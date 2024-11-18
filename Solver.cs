using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class Solver : MonoBehaviour
{
#region Parameters
    const int numHashes = 1<<20;
    const int numThreads = 1<<10; // Compute shader dependent value.
    public int numParticles = 1024;
    public float initSize = 10;
    public float radius = 1;
    public float gasConstant = 2000;
    public float restDensity = 10;
    public float mass = 1;
    public float density = 1;
    public float viscosity = 0.01f;
    public float gravity = 9.8f;
    public float deltaTime = 0.001f;

    public Vector3 minBounds = new Vector3(-10, -10, -10);
    public Vector3 maxBounds = new Vector3(10, 10, 10);

    public ComputeShader solverShader;

    public Shader renderShader;
    public Material renderMat;

    public Mesh particleMesh;
    public float particleRenderSize = 0.5f;

    public Mesh sphereMesh;

    public Color primaryColor;
    public Color secondaryColor;

    public Texture2D artTex;
    public bool bTwoCubes;

    private ComputeBuffer hashesBuffer;
    private ComputeBuffer globalHashCounterBuffer;
    private ComputeBuffer localIndicesBuffer;
    private ComputeBuffer inverseIndicesBuffer;
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer sortedBuffer;
    private ComputeBuffer forcesBuffer;
    private ComputeBuffer groupArrBuffer;
    private ComputeBuffer hashDebugBuffer;
    private ComputeBuffer hashValueDebugBuffer;
    private ComputeBuffer meanBuffer;
    private ComputeBuffer covBuffer;
    private ComputeBuffer principleBuffer;
    private ComputeBuffer hashRangeBuffer;

    private ComputeBuffer quadInstancedArgsBuffer;
    private ComputeBuffer sphereInstancedArgsBuffer;

    private int solverFrame = 0;

    private int moveParticleBeginIndex = 0;
    public int moveParticles = 10;

    private double lastFrameTimestamp = 0;
    private double totalFrameTime = 0;

    // @Temp: Just for fun.
    private int boundsState = 0;
    private float waveTime = 0;
    private Vector4[] boxPlanes = new Vector4[7];
    private Vector4[] wavePlanes = new Vector4[7];
    private Vector4[] groundPlanes = new Vector4[7];

    struct Particle {
        public Vector4 pos; // with pressure.
        public Vector4 vel; // vel
        public Vector4 col; // col
        // public Vector4 artCol;
    }

    private bool paused = false;
    private bool usePositionSmoothing = true;

    private CommandBuffer commandBuffer;
    private Mesh screenQuadMesh;

#endregion

    // HHH 点法式 计算平面方程 ax+by+cz+d=0，其中(a,b,c)是平面的法向量，|d|为原点到平面的距离
    Vector4 GetPlaneEq(Vector3 p, Vector3 n) {
        return new Vector4(n.x, n.y, n.z, -Vector3.Dot(p, n));
    }

    void UpdateParams() {
        if (Input.GetKeyDown(KeyCode.X)) {
            boundsState++;
        }

        Vector4[] currPlanes;
        switch (boundsState) {
            case 0: currPlanes = boxPlanes;
            break;

            case 1: currPlanes = wavePlanes;
            break;

            default: currPlanes = groundPlanes;
            break;
        }

        if (currPlanes == wavePlanes) {
            waveTime += deltaTime;
        }

        boxPlanes[0] = GetPlaneEq(new Vector3(0, 0, 0), Vector3.up);
        boxPlanes[1] = GetPlaneEq(new Vector3(0, 100, 0), Vector3.down);
        boxPlanes[2] = GetPlaneEq(new Vector3(-50, 0, 0), Vector3.right);
        boxPlanes[3] = GetPlaneEq(new Vector3(50, 0, 0), Vector3.left);
        boxPlanes[4] = GetPlaneEq(new Vector3(0, 0, -50), Vector3.forward);
        boxPlanes[5] = GetPlaneEq(new Vector3(0, 0, 50), Vector3.back);

        wavePlanes[0] = GetPlaneEq(new Vector3(0, 0, 0), Vector3.up);
        wavePlanes[1] = GetPlaneEq(new Vector3(0, 100, 0), Vector3.down);
        wavePlanes[2] = GetPlaneEq(new Vector3(-50 + Mathf.Pow(Mathf.Sin(waveTime*0.2f),8) * 25f, 0, 0), Vector3.right);
        wavePlanes[3] = GetPlaneEq(new Vector3(50, 0, 0), Vector3.left);
        wavePlanes[4] = GetPlaneEq(new Vector3(0, 0, -50), Vector3.forward);
        wavePlanes[5] = GetPlaneEq(new Vector3(0, 0, 50), Vector3.back);

        groundPlanes[0] = GetPlaneEq(new Vector3(0, 0, 0), Vector3.up);
        groundPlanes[1] = GetPlaneEq(new Vector3(0, 100, 0), Vector3.down);

        solverShader.SetVectorArray("planes", currPlanes);
    }

    void Start() {

        Particle[] particles = new Particle[numParticles];

        // Two dam break situation.
        Vector3 origin1 = new Vector3(
            Mathf.Lerp(minBounds.x, maxBounds.x, 0.25f),
            minBounds.y + initSize * 0.5f,
            Mathf.Lerp(minBounds.z, maxBounds.z, 0.25f)
        );
        Vector3 origin2 = new Vector3(
            Mathf.Lerp(minBounds.x, maxBounds.x, 0.75f),
            minBounds.y + initSize * 0.5f,
            Mathf.Lerp(minBounds.z, maxBounds.z, 0.75f)
        );
        Vector3 originFull = new Vector3(
            Mathf.Lerp(minBounds.x, maxBounds.x, 0.5f),
            minBounds.y + initSize * 0.5f,
            Mathf.Lerp(minBounds.z, maxBounds.z, 0.5f)
        );

        Color[] pixels = artTex.GetPixels();
        int artTexWidth = artTex.width;
        int artTexHeight = artTex.height;
        int artCnt = artTexWidth * artTexHeight;
        // HHH 计算texture中平均rgb
        Color avgCol = Color.clear;
        for(int i=0; i<artCnt;i++){
            avgCol.r += pixels[i].r;
            avgCol.g += pixels[i].g;
            avgCol.b += pixels[i].b;
        }
        avgCol.r /= artCnt;
        avgCol.g /= artCnt;
        avgCol.b /= artCnt;

        renderMat.SetColor("primaryColor", avgCol);

        for (int i = 0; i < numParticles; i++) {

            float randX = Random.Range(0.0f, 1f);
            float randY = Random.Range(0.0f, 1f);
            float randZ = Random.Range(0.0f, 1f);

            Vector3 pos = new Vector3(
                randX * initSize - initSize * 0.5f,
                randY * initSize - initSize * 0.5f,
                randZ * initSize - initSize * 0.5f
            );

            if( bTwoCubes )
                pos += (i % 2 == 0) ? origin1 : origin2;
            else
                pos += originFull;

            particles[i].pos = pos;
            int tmpId = (int)(artTexHeight*randZ)*artTexWidth + (int)(artTexWidth*randX);
            tmpId = tmpId > (artTexWidth*artTexWidth-1) ? (artTexWidth*artTexWidth-1):tmpId;
            Color tmpCol = pixels[tmpId];
            particles[i].vel = new Vector4(1.0f,1.0f,1.0f,1.0f);
            particles[i].col = new Vector4(tmpCol.r,tmpCol.g,tmpCol.b,1.0f);
            // particles[i].col = new Vector4(255.0f,255.0f,255.0f,255.0f);
        }



        solverShader.SetInt("numHash", numHashes);
        solverShader.SetInt("numParticles", numParticles);

        solverShader.SetFloat("radiusSqr", radius * radius);
        solverShader.SetFloat("radius", radius);
        solverShader.SetFloat("gasConst", gasConstant);
        solverShader.SetFloat("restDensity", restDensity);
        solverShader.SetFloat("mass", mass);
        solverShader.SetFloat("viscosity", viscosity);
        solverShader.SetFloat("gravity", gravity);
        solverShader.SetFloat("deltaTime", deltaTime);

        // HHH poly6 函数通常用于平滑粒子流体动力学（SPH）中，用于计算粒子之间的相互作用力
        float poly6 = 315f / (64f * Mathf.PI * Mathf.Pow(radius, 9f));
        // HHH spiky 函数通常用于平滑粒子流体动力学（SPH）中，用于计算粒子之间的相互作用力
        float spiky = 45f / (Mathf.PI * Mathf.Pow(radius, 6f));
        float visco = 45f / (Mathf.PI * Mathf.Pow(radius, 6f));

        solverShader.SetFloat("poly6Coeff", poly6);
        solverShader.SetFloat("spikyCoeff", spiky);
        solverShader.SetFloat("viscoCoeff", visco * viscosity);

        UpdateParams();

        hashesBuffer = new ComputeBuffer(numParticles, 4);

        globalHashCounterBuffer = new ComputeBuffer(numHashes, 4);

        localIndicesBuffer = new ComputeBuffer(numParticles, 4);

        inverseIndicesBuffer = new ComputeBuffer(numParticles, 4);

        // 1个浮点数4bit，每个particle有2-3个vector4 8 ---> 12
        particlesBuffer = new ComputeBuffer(numParticles, 4 * 12);
        particlesBuffer.SetData(particles);
        
        // 8 ---> 12
        sortedBuffer = new ComputeBuffer(numParticles, 4 * 12);

        forcesBuffer = new ComputeBuffer(numParticles * 2, 4 * 4);

        int groupArrLen = Mathf.CeilToInt(numHashes / 1024f);
        groupArrBuffer = new ComputeBuffer(groupArrLen, 4);

        hashDebugBuffer = new ComputeBuffer(4, 4);
        hashValueDebugBuffer = new ComputeBuffer(numParticles, 4 * 3);

        meanBuffer = new ComputeBuffer(numParticles, 4 * 4);
        covBuffer = new ComputeBuffer(numParticles * 2, 4 * 3);
        principleBuffer = new ComputeBuffer(numParticles * 4, 4 * 3);
        hashRangeBuffer = new ComputeBuffer(numHashes, 4 * 2);

        for (int i = 0; i < 13; i++) {
            solverShader.SetBuffer(i, "hashes", hashesBuffer);
            solverShader.SetBuffer(i, "globalHashCounter", globalHashCounterBuffer);
            solverShader.SetBuffer(i, "localIndices", localIndicesBuffer);
            solverShader.SetBuffer(i, "inverseIndices", inverseIndicesBuffer);
            solverShader.SetBuffer(i, "particles", particlesBuffer);
            solverShader.SetBuffer(i, "sorted", sortedBuffer);
            solverShader.SetBuffer(i, "forces", forcesBuffer);
            solverShader.SetBuffer(i, "groupArr", groupArrBuffer);
            solverShader.SetBuffer(i, "hashDebug", hashDebugBuffer);
            solverShader.SetBuffer(i, "mean", meanBuffer);
            solverShader.SetBuffer(i, "cov", covBuffer);
            solverShader.SetBuffer(i, "principle", principleBuffer);
            solverShader.SetBuffer(i, "hashRange", hashRangeBuffer);
            solverShader.SetBuffer(i, "hashValueDebug", hashValueDebugBuffer);
        }

        renderMat.SetBuffer("particles", particlesBuffer);
        renderMat.SetBuffer("principle", principleBuffer);
        renderMat.SetFloat("radius", particleRenderSize * 0.5f);

        quadInstancedArgsBuffer = new ComputeBuffer(1, sizeof(uint) * 5, ComputeBufferType.IndirectArguments);

        uint[] args = new uint[5];
        args[0] = particleMesh.GetIndexCount(0);
        args[1] = (uint) numParticles;
        args[2] = particleMesh.GetIndexStart(0);
        args[3] = particleMesh.GetBaseVertex(0);
        args[4] = 0;

        quadInstancedArgsBuffer.SetData(args);

        sphereInstancedArgsBuffer = new ComputeBuffer(1, sizeof(uint) * 5, ComputeBufferType.IndirectArguments);

        uint[] args2 = new uint[5];
        args2[0] = sphereMesh.GetIndexCount(0);
        args2[1] = (uint) numParticles;
        args2[2] = sphereMesh.GetIndexStart(0);
        args2[3] = sphereMesh.GetBaseVertex(0);
        args2[4] = 0;

        sphereInstancedArgsBuffer.SetData(args2);

        screenQuadMesh = new Mesh();
        screenQuadMesh.vertices = new Vector3[4] {
            new Vector3( 1.0f , 1.0f,  0.0f),
            new Vector3(-1.0f , 1.0f,  0.0f),
            new Vector3(-1.0f ,-1.0f,  0.0f),
            new Vector3( 1.0f ,-1.0f,  0.0f),
        };
        screenQuadMesh.uv = new Vector2[4] {
            new Vector2(1, 0),
            new Vector2(0, 0),
            new Vector2(0, 1),
            new Vector2(1, 1)
        };
        screenQuadMesh.triangles = new int[6] { 0, 1, 2, 2, 3, 0 };

        commandBuffer = new CommandBuffer();
        commandBuffer.name = "Fluid Render";

        UpdateCommandBuffer();
        Camera.main.AddCommandBuffer(CameraEvent.AfterForwardAlpha, commandBuffer);

    }

    void Update() {
        // Update solver.
        {
            UpdateParams();

            if (Input.GetMouseButton(0)) {
                Ray mouseRay = Camera.main.ScreenPointToRay(Input.mousePosition);
                RaycastHit hit;
                if (Physics.Raycast(mouseRay, out hit)) {
                    Vector3 pos = new Vector3(
                        Mathf.Clamp(hit.point.x, minBounds.x, maxBounds.x),
                        maxBounds.y - 1f,
                        Mathf.Clamp(hit.point.z, minBounds.z, maxBounds.z)
                    );

                    solverShader.SetInt("moveBeginIndex", moveParticleBeginIndex);
                    solverShader.SetInt("moveSize", moveParticles);
                    solverShader.SetVector("movePos", pos);
                    solverShader.SetVector("moveVel", Vector3.down * 70);

                    solverShader.Dispatch(solverShader.FindKernel("MoveParticles"), 1, 1, 1);

                    moveParticleBeginIndex = (moveParticleBeginIndex + moveParticles * moveParticles) % numParticles;
                }
            }

            if (Input.GetKeyDown(KeyCode.Space)) {
                paused = !paused;
            }

            if (Input.GetKeyDown(KeyCode.Z)) {
                usePositionSmoothing = !usePositionSmoothing;
                Debug.Log("usePositionSmoothing: " + usePositionSmoothing);
            }

            renderMat.SetColor("primaryColor", primaryColor.linear);
            renderMat.SetColor("secondaryColor", secondaryColor.linear);
            renderMat.SetInt("usePositionSmoothing", usePositionSmoothing ? 1 : 0);

            double solverStart = Time.realtimeSinceStartupAsDouble;

            solverShader.Dispatch(solverShader.FindKernel("ResetCounter"), Mathf.CeilToInt((float)numHashes / numThreads), 1, 1);
            solverShader.Dispatch(solverShader.FindKernel("InsertToBucket"), Mathf.CeilToInt((float)numParticles / numThreads), 1, 1);

            // Debug
            if (Input.GetKeyDown(KeyCode.C)) {
                uint[] debugResult = new uint[4];

                hashDebugBuffer.SetData(debugResult);

                solverShader.Dispatch(solverShader.FindKernel("DebugHash"), Mathf.CeilToInt((float)numHashes / numThreads), 1, 1);

                hashDebugBuffer.GetData(debugResult);

                uint usedHashBuckets = debugResult[0];
                uint maxSameHash = debugResult[1];

                Debug.Log($"Total buckets: {numHashes}, Used buckets: {usedHashBuckets}, Used rate: {(float)usedHashBuckets / numHashes * 100}%");
                Debug.Log($"Avg hash collision: {(float)numParticles / usedHashBuckets}, Max hash collision: {maxSameHash}");
            }

            solverShader.Dispatch(solverShader.FindKernel("PrefixSum1"), Mathf.CeilToInt((float)numHashes / numThreads), 1, 1);

            // @Important: Because of the way prefix sum algorithm implemented,
            // Currently maximum numHashes value is numThreads^2.
            Debug.Assert(numHashes <= numThreads*numThreads);
            solverShader.Dispatch(solverShader.FindKernel("PrefixSum2"), 1, 1, 1);

            solverShader.Dispatch(solverShader.FindKernel("PrefixSum3"), Mathf.CeilToInt((float)numHashes / numThreads), 1, 1);
            solverShader.Dispatch(solverShader.FindKernel("Sort"), Mathf.CeilToInt((float)numParticles / numThreads), 1, 1);
            solverShader.Dispatch(solverShader.FindKernel("CalcHashRange"), Mathf.CeilToInt((float)numHashes / numThreads), 1, 1);

            // Debug
            if (Input.GetKeyDown(KeyCode.C)) {
                uint[] debugResult = new uint[4];

                int[] values = new int[numParticles * 3];

                hashDebugBuffer.SetData(debugResult);

                solverShader.Dispatch(solverShader.FindKernel("DebugHash"), Mathf.CeilToInt((float)numHashes / numThreads), 1, 1);

                hashDebugBuffer.GetData(debugResult);

                uint totalAccessCount = debugResult[2];
                uint totalNeighborCount = debugResult[3];

                Debug.Log($"Total access: {totalAccessCount}, Avg access: {(float)totalAccessCount / numParticles}, Avg accept: {(float)totalNeighborCount / numParticles}");
                Debug.Log($"Average accept rate: {(float)totalNeighborCount / totalAccessCount * 100}%");

                hashValueDebugBuffer.GetData(values);

                HashSet<Vector3Int> set = new HashSet<Vector3Int>();
                for (int i = 0; i < numParticles; i++) {
                    Vector3Int vi = new Vector3Int(values[i*3+0], values[i*3+1], values[i*3+2]);
                    set.Add(vi);
                }

                Debug.Log($"Total unique hash keys: {set.Count}, Ideal bucket load: {(float)set.Count / numHashes * 100}%");
            }

            if (!paused) {
                for (int iter = 0; iter < 1; iter++) {
                    solverShader.Dispatch(solverShader.FindKernel("CalcPressure"), Mathf.CeilToInt((float)numParticles / 128), 1, 1);
                    solverShader.Dispatch(solverShader.FindKernel("CalcForces"), Mathf.CeilToInt((float)numParticles / 128), 1, 1);
                    solverShader.Dispatch(solverShader.FindKernel("CalcPCA"), Mathf.CeilToInt((float)numParticles / numThreads), 1, 1);
                    solverShader.Dispatch(solverShader.FindKernel("Step"), Mathf.CeilToInt((float)numParticles / numThreads), 1, 1);
                }

                solverFrame++;

                if (solverFrame > 1) {
                    totalFrameTime += Time.realtimeSinceStartupAsDouble - lastFrameTimestamp;
                }

                if (solverFrame == 400 || solverFrame == 1200) {
                    Debug.Log($"Avg frame time at #{solverFrame}: {totalFrameTime / (solverFrame-1) * 1000}ms.");
                }
            }

            lastFrameTimestamp = Time.realtimeSinceStartupAsDouble;
        }
    }

    void UpdateCommandBuffer() {
        commandBuffer.Clear();

        int[] worldPosBufferIds = new int[] {
            Shader.PropertyToID("worldPosBuffer1"),
            Shader.PropertyToID("worldPosBuffer2")
        };

        // 创建一个存位置的渲染纹理
        commandBuffer.GetTemporaryRT(worldPosBufferIds[0], Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBFloat);
        // commandBuffer.GetTemporaryRT(worldPosBufferIds[1], Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBFloat);

        // 创建一个存深度的渲染纹理
        // 可以使用 depthId 来设置深度缓冲区作为渲染目标，这样所有的深度信息都会被渲染到这个指定的缓冲区中
        int depthId = Shader.PropertyToID("depthBuffer");
        commandBuffer.GetTemporaryRT(depthId, Screen.width, Screen.height, 32, FilterMode.Point, RenderTextureFormat.Depth);

        // 设置渲染对象为位置纹理和深度纹理
        // SetRenderTarget 方法用于指定后续渲染命令的目标纹理，即告诉 GPU 将渲染的结果输出到哪个或哪些渲染纹理中
        commandBuffer.SetRenderTarget((RenderTargetIdentifier)worldPosBufferIds[0], (RenderTargetIdentifier)depthId);
        // QQQ 把 clearDepth 和 clearColor 的两个 flag 设置为 true，清除缓冲区，保证不会受到之前内容的影响
        commandBuffer.ClearRenderTarget(true, true, Color.clear);


        // 1.
        // 借助ComputerShader实现超过1024的Instance绘制 用 Pass0
        // QQQ 通过间接实例化绘制网格；这个方法用于在命令缓冲区中绘制实例化网格，适用于需要高效渲染多个相同网格的场景
        commandBuffer.DrawMeshInstancedIndirect(
            sphereMesh,
            0,  // submeshIndex
            renderMat,
            0,  // shaderPass
            sphereInstancedArgsBuffer
        );


        // 创建一个存深度的渲染纹理2
        int depth2Id = Shader.PropertyToID("depth2Buffer");
        commandBuffer.GetTemporaryRT(depth2Id, Screen.width, Screen.height, 32, FilterMode.Point, RenderTextureFormat.Depth);

        commandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
        // HHH 该函数 设定了 以下的 Draw 的着色器 将颜色输出到 第一个变量，将深度输出到 第二个变量
        commandBuffer.SetRenderTarget((RenderTargetIdentifier)worldPosBufferIds[0], (RenderTargetIdentifier)depth2Id);

        /// !!!
        /// 以下是QQQ增加的代码
        commandBuffer.ClearRenderTarget(true, true, Color.clear);
        commandBuffer.SetGlobalTexture("depthBuffer", depthId);
        // 设置 pass0 中得到的深度渲染结果（屏幕空间的像素深度）为全局纹理，供 shader 后续的 pass 使用
        /// end

        // 2.
        // 绘制一个面，用Pass1
        commandBuffer.DrawMesh(
            screenQuadMesh, // QQQ 是代码前面定义好的一个屏幕四边形
            Matrix4x4.identity,
            renderMat,
            0, // submeshIndex
            1  // shaderPass
        );

        int normalBufferId = Shader.PropertyToID("normalBuffer");
        commandBuffer.GetTemporaryRT(normalBufferId, Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

        int colorBufferId = Shader.PropertyToID("colorBuffer");
        commandBuffer.GetTemporaryRT(colorBufferId, Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.RGHalf);

        int oriColorBufferId = Shader.PropertyToID("oriColorBuffer");
        commandBuffer.GetTemporaryRT(oriColorBufferId, Screen.width, Screen.height, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);

        // HHH 设置以下 Draw 渲染输出目标
        commandBuffer.SetRenderTarget(new RenderTargetIdentifier[] { normalBufferId, colorBufferId, oriColorBufferId }, (RenderTargetIdentifier)depth2Id);
        // QQQ 清理缓冲区，但是注意这里设置了不清理深度缓冲区值（第一个false）
        commandBuffer.ClearRenderTarget(false, true, Color.clear);
        // QQQ 将 pass1 新计算好的像素的世界坐标设置为全局纹理 worldPosBuffer 供 pass2 调用
        commandBuffer.SetGlobalTexture("worldPosBuffer", worldPosBufferIds[0]);

        // 3.
        // particleMesh是一个立方体，估计是计算法线不用那么准确
        // HHH 计算其法线，好像是可以用其包围盒来简化计算的 用 Pass 2
        commandBuffer.DrawMeshInstancedIndirect(
            particleMesh,
            0,  // submeshIndex
            renderMat,
            2,  // shaderPass
            quadInstancedArgsBuffer
        );

        // QQQ 将 pass2 计算出的 normalBuffer colorBuffer oriColorBuffer 设置为全局纹理供 pass3 使用
        commandBuffer.SetGlobalTexture("normalBuffer", normalBufferId);
        commandBuffer.SetGlobalTexture("colorBuffer", colorBufferId);
        commandBuffer.SetGlobalTexture("oriColorBuffer", oriColorBufferId);
        // QQQ 设置 pass3 的渲染目标为屏幕
        commandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);

        // 4.
        // 绘制一个面，用Pass 3, 这里做表面的光照渲染
        commandBuffer.DrawMesh(
            screenQuadMesh,
            Matrix4x4.identity,
            renderMat,
            0, // submeshIndex
            3  // shaderPass
        );
    }

    // HHH
    // LateUpdate 方法在 Unity 的每一帧渲染结束时调用
    void LateUpdate() {
        Matrix4x4 view = Camera.main.worldToCameraMatrix;

        // HHH
        // 将计算得到的 viewMatrix 的逆矩阵设置为全局着色器变量 inverseV。这样，在着色器中就可以访问这个矩阵，用于将摄像机坐标转换为世界坐标。
        Shader.SetGlobalMatrix("inverseV", view.inverse);
        // 将主摄像机的投影矩阵的逆矩阵设置为全局着色器变量 inverseP。这个逆矩阵在着色器中用于将裁剪空间坐标转换为摄像机空间坐标。
        Shader.SetGlobalMatrix("inverseP", Camera.main.projectionMatrix.inverse);
    }

    void OnDisable() {
        hashesBuffer.Dispose();
        globalHashCounterBuffer.Dispose();
        localIndicesBuffer.Dispose();
        inverseIndicesBuffer.Dispose();
        particlesBuffer.Dispose();
        sortedBuffer.Dispose();
        forcesBuffer.Dispose();
        groupArrBuffer.Dispose();
        hashDebugBuffer.Dispose();
        meanBuffer.Dispose();
        covBuffer.Dispose();
        principleBuffer.Dispose();
        hashRangeBuffer.Dispose();

        quadInstancedArgsBuffer.Dispose();
    }
}
