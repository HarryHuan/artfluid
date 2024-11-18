Shader "Spheres"
{
    Properties
    {
        _PrimaryColor ("Primary Color", Color) = (1,1,1,1)
        _SecondaryColor ("Secondary Color", Color) = (1,1,1,1)
        _FoamColor ("Foam Color", Color) = (1,1,1,1)
        // HHH HDR high dynamic range 高动态区域 使用32位而不是8位表示亮度 亮的更亮 
        [HDR] _SpecularColor ("Specular Color", Color) = (1,1,1,1)
        _PhongExponent ("Phong Exponent", Float) = 128
        _EnvMap ("Environment Map", Cube) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        // Pass 0
        // HHH 将顶点的屏幕空间深度写入到 depthBuffer
        Pass
        {
            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct Particle {
                float4 pos;
                float4 vel;
                float4 col;
            };

            StructuredBuffer<Particle> particles;

            float radius;

            StructuredBuffer<float3> principle; // QQQ 粒子形状和方向信息

            int usePositionSmoothing; // QQQ 是否平滑处理粒子位置

            // QQQ 定义顶点着色器的输入结构体
            struct appdata
            {
                float4 vertex : POSITION; // QQQ 顶点的模型空间位置
            };

            struct v2f
            {
                float4 vertex : SV_POSITION; // QQQ 裁剪空间的顶点位置
            };

            // https://www.iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
            // HHH 求光线与球体的交点
            float sphIntersect( float3 ro, float3 rd, float4 sph )
            {
                float3 oc = ro - sph.xyz;
                float b = dot( oc, rd );
                float c = dot( oc, oc ) - sph.w*sph.w;
                float h = b*b - c;
                if( h<0.0 ) return -1.0;
                h = sqrt( h );
                return -b - h;
            }

            float invlerp(float a, float b, float t) {
                return (t-a)/(b-a);
            }

            v2f vert (appdata v, uint id : SV_InstanceID)
            {
                // QQQ 表示世界坐标的粒子中心位置
                // 根据平滑与否分别取 principle 里储存的平滑处理后的粒子位置 or particles 里面存储的 pos 坐标
                float3 spherePos = usePositionSmoothing ? principle[id*4+3] : particles[id].pos.xyz;
                // QQQ 输入 v 是模型空间的坐标，基于粒子半径 radius 进行粒子形状的缩放，得到 localPos
                float3 localPos = v.vertex.xyz * (radius * 2 * 2);
                // QQQ 矩阵，表示单个粒子的方向和拉伸形变信息。
                // 这一步是拷贝出 principle 里记录的粒子形状和方向信息，用来后续对粒子施加椭圆形变
                float3x3 ellip = float3x3(principle[id*4+0], principle[id*4+1], principle[id*4+2]);
                // QQQ 把 ellip 和 localPos 进行矩阵乘，即对粒子施加椭圆形变，并加上世界空间中的粒子中心位置得到世界坐标
                float3 worldPos = mul(ellip, localPos) + spherePos;

                v2f o;
                // o.vertex.w是深度值
                // VP * 世界坐标 = 裁剪空间 [-1,1]^3
                o.vertex = mul(UNITY_MATRIX_VP, float4(worldPos, 1));
                return o;
            }

            // QQQ PassO 片元着色器不渲染任何颜色
            fixed4 frag (v2f i) : SV_Target
            {
                return 0;   
                // Depth: return float4(i.vertex.www*0.004, 1.0); //
            }
            ENDCG
        }

        // Pass 1
        // QQQ 将计算出来的像素的世界坐标写入 worldPosBufferIds[0]
        // QQQ 将从 Pass0 读取的深度输出到 depth2Id
        Pass
        {
            // HHH 控制深度测试，决定如何根据深度信息来拒绝或接受像素
            // HHH Always：不进行深度测试。绘制所有几何体，无论距离如何
            ZTest Always

            // HHH CG/HLSL 代码块：包含实际的着色器代码，定义顶点着色器和片元着色器的逻辑
            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D depthBuffer; // QQQ Solver.cs 中设置的全局 texture，也就是 Pass0 计算结果

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            float4x4 inverseV, inverseP;

            float radius;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = v.vertex;
                o.vertex.z = 0.5;
                o.uv = v.uv;
                return o;
            }

            // QQQ 将计算出来的像素的世界坐标写入 worldPosBufferIds[0]
            // QQQ 将从 Pass0 读取的深度输出到 depth2Id
            // HHH SV_Depth 是一个语义，指定这个变量应该绑定到片元着色器的深度输出，这是深度缓冲区的常用绑定点
            // HHH SV_Target 是一个语义，用于指定这个变量是渲染目标的颜色输出
            float4 frag(v2f i, out float depth : SV_Depth) : SV_Target
            {
                // HHH 深度值是实际线性距离沿着-z方向的分量距离
                float d = tex2D(depthBuffer, i.uv);

                depth = d;

                // Calculate world-space position.
                // HHH i.uv*2-1：这将屏幕空间中的 UV 坐标（范围从 [0, 1]）转换到 [-1, 1] 的范围，这对应于屏幕空间中的 [-1, 1]
                // QQQ 利用投影矩阵的逆，得到每个像素在观察空间的位置
                // 归一化，得到 摄像机观察屏幕上该像素的方向向量
                float3 viewSpaceRayDir = normalize(mul(inverseP, float4(i.uv*2-1, 0, 1)).xyz);
                // HHH 计算从相机到深度纹理中给定点的线性距离
                // QQQ 线性深度 / 余弦值 = 距离
                float viewSpaceDistance = LinearEyeDepth(d) / dot(viewSpaceRayDir, float3(0,0,-1));
                // Slightly push forward to screen.
                // viewSpaceDistance -= radius * 1;
                // viewSpaceDistance -= 0.1;

                // QQQ 从摄像机到像素点的方向单位向量 viewSpaceRayDir * 两者距离 viewSpaceDistance = 从摄像机到像素点的向量
                // 即 像素在观察空间的坐标
                float3 viewSpacePos = viewSpaceRayDir * viewSpaceDistance;
                // 视图矩阵的逆 * 观察坐标 得到世界坐标，补 1 成为齐次坐标
                float3 worldSpacePos = mul(inverseV, float4(viewSpacePos, 1)).xyz;
                // 补 0 表示为向量
                return float4(worldSpacePos, 0);
            }

            ENDCG
        }

        // Pass 2
        // HHH 将像素的法线和权重（由离粒子中心的距离决定）输出到 normalBuffer
        // HHH 将像素的带权重的速度输出到 colorBuffer
        // HHH 将指定的渲染纹理图在该像素位置的rgb绑定权重输出到 oriColorBuffer
        Pass
        {
            // HHH Less：绘制位于现有几何体前面的几何体。不绘制位于现有几何体相同距离或后面的几何体
            ZTest Less
            // HHH ZWrite 参数用于控制是否将深度信息写入深度缓冲区
            // HHH Off 禁用写入深度缓冲区。这意味着片段的深度值将不会被写入深度缓冲区
            ZWrite Off
            // HHH SrcBlend 和 DstBlend 都设置为 One 
            // 源颜色和目标颜色直接相加，可能会造成颜色值超出正常范围（0-1），从而产生过曝或饱和的效果
            Blend One One

            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct Particle {
                float4 pos;
                float4 vel;
                float4 col;
            };

            StructuredBuffer<Particle> particles;

            float radius;

            StructuredBuffer<float3> principle;

            int usePositionSmoothing;

            // QQQ 由 Pass1 新计算的像素的世界坐标，由 solver 将其设置为全局纹理
            sampler2D worldPosBuffer;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                // HHH xxx : XXX 将 xxx 参数绑定到 XXX unity中的渲染固定声明的变量
                // HHH 通过SV_POSITION语义来将一个float4类型的变量标记为顶点的在齐次裁剪空间下的坐标信息
                float4 vertex : SV_POSITION; // 裁剪空间中的顶点坐标
                float4 rayDir : TEXCOORD0; // 从相机到像素方向的归一化向量
                float3 rayOrigin: TEXCOORD1; // 光线的起点位置
                float4 spherePos : TEXCOORD2; // 存储球体的位置，球心坐标和w半径
                float2 densitySpeed : TEXCOORD3; // QQQ 归一化之后的粒子密度和总速度模长
                float3 m1 : TEXCOORD4; // QQQ 粒子的椭圆型变逆矩阵的列向量
                float3 m2 : TEXCOORD5;
                float3 m3 : TEXCOORD6;
                float3 oriColor : TEXCOORD7;
            };

            struct output2
            {
                float4 normal : SV_Target0; // 输出到 normalBuffer
                float2 densitySpeed : SV_Target1; // 输出到 colorBuffer
                float4 oriColor : SV_Target2; // 输出到 oriColorBuffer
            };

            // https://www.iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
            float sphIntersect( float3 ro, float3 rd, float4 sph )
            {
                float3 oc = ro - sph.xyz;
                float b = dot( oc, rd );
                float c = dot( oc, oc ) - sph.w*sph.w;
                float h = b*b - c;
                if( h<0.0 ) return -1.0;
                h = sqrt( h );
                return -b - h;
            }

            float invlerp(float a, float b, float t) {
                return (t-a)/(b-a);
            }
            /// HHH 求矩阵逆 A⁻¹=A*/|A|
            float3x3 inverse(float3x3 m) {
                float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
                float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
                float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

                float b01 = a22 * a11 - a12 * a21;
                float b11 = -a22 * a10 + a12 * a20;
                float b21 = a21 * a10 - a11 * a20;

                float det = a00 * b01 + a01 * b11 + a02 * b21;
                return float3x3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11),
                            b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
                            b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det;
            }

            v2f vert (appdata v, uint id : SV_InstanceID)
            {
                float3 spherePos = usePositionSmoothing ? principle[id*4+3] : particles[id].pos.xyz;
                float3 localPos = v.vertex.xyz * (radius * 2 * 4);

                float3x3 ellip = float3x3(principle[id*4+0], principle[id*4+1], principle[id*4+2]);

                float3 worldPos = mul(ellip, localPos) + spherePos;
                // QQQ 以上 计算运动带来的形变，把粒子坐标从模型空间转换到了世界空间

                ellip = inverse(ellip);
                // QQQ 相减，得到相机相对于球体的位置
                float3 objectSpaceCamera = _WorldSpaceCameraPos.xyz - spherePos;
                // QQQ 再与形变逆矩阵相乘，把相机相对于球体的位置转换到模型空间
                objectSpaceCamera = mul(ellip, objectSpaceCamera);
                // HHH 从世界空间中相机到顶点位置的方向向量
                float3 objectSpaceDir = normalize(worldPos - _WorldSpaceCameraPos.xyz);
                // HHH 同上，将其转到模型空间
                objectSpaceDir = mul(ellip, objectSpaceDir);
                objectSpaceDir = normalize(objectSpaceDir);

                v2f o;
                o.vertex = mul(UNITY_MATRIX_VP, float4(worldPos, 1));

                // @Temp: Actually it's screen-space uv.
                // GPU 自实现
                // QQQ 顶点的屏幕空间坐标，由顶点的裁剪空间坐标用内置函数计算出，[0,1]
                o.rayDir = ComputeScreenPos(o.vertex);
                // QQQ 把相机在模型空间相对球体的位置赋值给 rayOrigin
                o.rayOrigin = objectSpaceCamera;
                // QQQ 加入的 particles[id].pos.w 表示粒子的密度信息
                o.spherePos = float4(spherePos, particles[id].pos.w); // Add density values.

                // @Hardcoded: Range
                // QQQ length(particles[id].vel.xyz) 是粒子三个方向速度的模长，估算整体的速度大小；
                // 然后把粒子的密度 o.spherePos.w 和速度模长通过反向插值归一化，再通过 saturate 限定到 [0.1]
                o.densitySpeed = saturate(float2(invlerp(0, 1, o.spherePos.w), invlerp(10, 30, length(particles[id].vel.xyz))));
                // o.densitySpeed = saturate(float2(invlerp(0, 1, o.spherePos.w), invlerp(10, 30, 1)));
                // HHH particles[id].col.xyz 这是用来渲染的纹理颜色rgb三通量，与粒子的密度信息 打包给 oriColorBuffer
                o.oriColor = float4( particles[id].col.xyz, particles[id].pos.w );

                o.m1 = ellip._11_12_13;
                o.m2 = ellip._21_22_23;
                o.m3 = ellip._31_32_33;
                return o;
            }

            output2 frag (v2f i) : SV_Target
            {
                // QQQ 拼接椭圆变换矩阵的逆矩阵
                float3x3 mInv = float3x3(i.m1, i.m2, i.m3);
                // QQQ 把顶点屏幕坐标的 xy 除以 w 分量，转换为二维 uv 坐标
                float2 uv = i.rayDir.xy / i.rayDir.w;
                // QQQ 通过 uv 坐标从像素世界坐标 worldPosBuffer(from pass1) 中采样，得到当前像素的世界坐标
                float3 worldPos = tex2D(worldPosBuffer, uv).xyz;
                // QQQ worldPos - i.spherePos.xyz像素相对球体中心位置的偏移，然后用椭圆形变的逆矩阵变换，得到在模型空间的偏移量
                float3 ellipPos = mul(mInv, worldPos - i.spherePos.xyz);

                float distSqr = dot(ellipPos, ellipPos);
                float radiusSqr = pow(radius*4, 2);
                // QQQ 对比两者，如果球体半径平方更小，说明这个像素在球体外部，不需要再渲染，直接跳过
                if (distSqr >= radiusSqr) discard;

                // QQQ 法线和普通位置向量的变换不同，法线是一个垂直于表面的向量。
                // 在一个如椭圆变换的非均匀缩放后，法线的变换需要使用矩阵的逆转置，来保证法线在缩放时被正确变换
                mInv = mul(transpose(mInv), mInv);
                // QQQ 计算一个权重，离粒子中心越近，权重越大
                float weight = pow(1 - distSqr / radiusSqr, 3);
                // QQQ 像素在世界坐标下和粒子中心的偏移
                float3 centered = worldPos - i.spherePos.xyz;
                // QQQ 计算偏移向量在椭圆变换下的梯度
                float3 grad = mul(mInv, centered) + mul(centered, mInv);
                // QQQ 计算得到新的法线方向
                float3 normal = grad * weight;

                output2 o;
                // QQQ 将法线和权重打包输出到 noramlBuffer
                o.normal = float4(normal, weight);
                // QQQ 计算密度输出到 colorBuffer
                o.densitySpeed = float2(i.densitySpeed) * weight;
                // QQQ 把原 oriColor 再打包一个权重
                o.oriColor = float4(i.oriColor, weight);
                return o;
            }
            ENDCG
        }

        // Pass 3
        // HHH 
        // 目的：渲染并绘制帧。
        // 顶点着色器 (vert)：将顶点位置设置为固定的深度值（0.5），并传递 UV 坐标。
        // 片元着色器 (frag)：从多个缓冲区采样数据，计算光照和反射，然后输出最终的颜色。
        Pass
        {
            CGPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D depthBuffer; // pass0 计算的深度信息
            sampler2D worldPosBuffer; // pass1 计算的像素世界坐标
            sampler2D normalBuffer; // pass2 计算的像素法线和权重信息
            sampler2D colorBuffer; // pass2 计算的像素加权速度信息
            sampler2D oriColorBuffer; // pass2 被鼠标指定的贴图rgb信息以及该点的权重信息
            samplerCUBE _EnvMap;

            float4 _PrimaryColor, _SecondaryColor, _FoamColor;
            float4 _SpecularColor;
            float _PhongExponent;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = v.vertex;
                o.vertex.z = 0.5;
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i, out float depth : SV_Depth) : SV_Target
            {
                float d = tex2D(depthBuffer, i.uv);
                // HHH 
                // By storing world positions in a texture, the shader can efficiently access this information 
                // during the lighting pass to perform calculations that depend on the object's position in the world.
                float3 worldPos = tex2D(worldPosBuffer, i.uv).xyz;
                float4 normal = tex2D(normalBuffer, i.uv);
                float2 densitySpeed = tex2D(colorBuffer, i.uv);

                if (d == 0) discard;

                if (normal.w > 0) {
                    normal.xyz = normalize(normal.xyz);
                    densitySpeed /= normal.w;
                }

                depth = d;

                // 从一张纯黄的纹理里采颜色，居然会采到黑色，特别是粒子飞溅的区域
                // 颜色没错，应该是稀疏的地方点少，然后叠加的就少。密度高的地方乘以的系数应该要小，密度低的地方乘以的系数要高
                float4 oriCol = tex2D(oriColorBuffer, i.uv);
                float tmpDensity = 0.1f;
                if ( oriCol.w > 0.0f )
                    tmpDensity = 1.0f / oriCol.w * 0.12f;//sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(oriCol.w)))))) * 0.05;
                else
                    tmpDensity = 0.6f;
                if( tmpDensity > 0.6f )
                    tmpDensity = 0.6f;
                // HHH 计算漫反射
                float3 diffuse = oriCol.xyz * tmpDensity;//lerp(oriCol, _SecondaryColor, densitySpeed.x);
                diffuse = lerp(_PrimaryColor, diffuse, densitySpeed.x);
                diffuse = lerp(diffuse, _FoamColor, densitySpeed.y);
                // HHH 计算反射光照
                float light = max(dot(normal, _WorldSpaceLightPos0.xyz), 0);
                light = lerp(0.1, 1, light);

                float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - worldPos);
                float3 lightDir = _WorldSpaceLightPos0.xyz;
                // QQQ 视线、光照方向的归一化和，就是 Blinn-Phong 模型的半向量
                float3 mid = normalize(viewDir + lightDir);

                // Specular highlight 高光
                diffuse += pow(max(dot(normal, mid), 0), _PhongExponent) * _SpecularColor;
                // QQQ 环境反射光，reflect(-viewDir, normal) 计算反射向量，然后通过 textCUBE 从环境立方体贴图中采样反射颜色
                float4 reflectedColor = texCUBE(_EnvMap, reflect(-viewDir, normal));

                // Schlick's approximation, 菲涅耳的 Schlick 近似
                // 计算反射的折射率
                float iorAir = 1.0;
                float iorWater = 1.333;
                float r0 = pow((iorAir - iorWater) / (iorAir + iorWater), 2);
                float rTheta = r0 + (1 - r0) * pow(1 - max(dot(viewDir, normal), 0), 5);
                diffuse = lerp(diffuse, reflectedColor, rTheta);
                
                return float4(diffuse.xyz, 1);
            }

            ENDCG
        }
    }
}
