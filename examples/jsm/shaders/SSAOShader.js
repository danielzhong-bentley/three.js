import {
	Matrix4,
	Vector2
} from 'three';

/**
 * References:
 * http://john-chapman-graphics.blogspot.com/2013/01/ssao-tutorial.html
 * https://learnopengl.com/Advanced-Lighting/SSAO
 * https://github.com/McNopper/OpenGL/blob/master/Example28/shader/ssao.frag.glsl
 */

const SSAOShader = {

	name: 'SSAOShader',

	defines: {
		'PERSPECTIVE_CAMERA': 1,
		'KERNEL_SIZE': 32
	},

	uniforms: {

		'tNormal': { value: null },
		'tDepth': { value: null },
		'tNoise': { value: null },
		'kernel': { value: null },
		'cameraNear': { value: null },
		'cameraFar': { value: null },
		'resolution': { value: new Vector2() },
		'cameraProjectionMatrix': { value: new Matrix4() },
		'cameraInverseProjectionMatrix': { value: new Matrix4() },
		'cameraInverseViewMatrix': { value: new Matrix4() },
		'cameraViewMatrix': { value: new Matrix4() },
		'kernelRadius': { value: 8 },
		'minDistance': { value: 0.005 },
		'maxDistance': { value: 0.05 },

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;

			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`
		uniform highp sampler2D tNormal;
		uniform highp sampler2D tDepth;
		uniform sampler2D tNoise;

		uniform vec3 kernel[ KERNEL_SIZE ];

		uniform vec2 resolution;

		uniform float cameraNear;
		uniform float cameraFar;
		uniform mat4 cameraProjectionMatrix;
		uniform mat4 cameraInverseProjectionMatrix;
		uniform mat4 cameraInverseViewMatrix;
		uniform mat4 cameraViewMatrix;

		uniform float kernelRadius;
		uniform float minDistance; // avoid artifacts caused by neighbour fragments with minimal depth difference
		uniform float maxDistance; // avoid the influence of fragments which are too far away
		// uniform vec3 cameraPosition;

		varying vec2 vUv;

		#include <packing>

		float getDepth( const in vec2 screenPosition ) {

			return texture2D( tDepth, screenPosition ).x;

		}

		float getLinearDepth( const in vec2 screenPosition ) {
			// return texture2D( tDepth, screenPosition ).x;

			#if PERSPECTIVE_CAMERA == 1

				float fragCoordZ = texture2D( tDepth, screenPosition ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );

			#else

				return texture2D( tDepth, screenPosition ).x;

			#endif

		}

		float getViewZ( const in float depth ) {

			#if PERSPECTIVE_CAMERA == 1

				return perspectiveDepthToViewZ( depth, cameraNear, cameraFar );

			#else

				return orthographicDepthToViewZ( depth, cameraNear, cameraFar );

			#endif

		}

		vec3 getViewPosition( const in vec2 screenPosition, const in float depth, const in float viewZ ) {

			float clipW = cameraProjectionMatrix[2][3] * viewZ + cameraProjectionMatrix[3][3];

			vec4 clipPosition = vec4( ( vec3( screenPosition, depth ) - 0.5 ) * 2.0, 1.0 );

			clipPosition *= clipW; // unprojection.

			return ( cameraInverseProjectionMatrix * clipPosition ).xyz;

		}

		vec3 getViewNormal( const in vec2 screenPosition ) {

			return unpackRGBToNormal( texture2D( tNormal, screenPosition ).xyz );

		}
			
		vec3 getWorldPosition( const in vec3 viewPosition ) {
			vec4 viewPos4 = vec4(viewPosition, 1.0);

			vec4 worldPosition = cameraInverseViewMatrix * viewPos4;

			return worldPosition.xyz;
		}

		vec3 getWorldNormal( const in vec3 viewNormal ) {
			return normalize((cameraInverseViewMatrix * vec4(viewNormal, 0.0)).xyz);
		}

		void main() {

			float depth = getDepth( vUv );

			if ( depth == 1.0 ) {

				// gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
				gl_FragColor = vec4( 1.0 ); // don't influence background
				
			} else {

				float viewZ = getViewZ( depth );
				
				vec3 viewPosition = getViewPosition( vUv, depth, viewZ );
				
				vec3 viewNormal = getViewNormal( vUv );

				// gl_FragColor = vec4(0.5 + 0.5 * viewNormal, 1.0);
								
				// Daniel Zhong's code
				vec3 worldPosition = getWorldPosition(viewPosition);
				vec3 worldNormal = getWorldNormal(viewNormal);
				
				float Sx = (float(kernelRadius) / 2.0) / resolution.x;
        		float Sy = (float(kernelRadius) / 2.0) / resolution.y;
				// float worldSpaceZ = length(worldPosition - cameraPosition);
		
				float worldSpaceZ = dot(worldPosition - cameraPosition, -cameraInverseViewMatrix[2].xyz);
				float kernelDiagonal = sqrt(Sx * Sx + Sy * Sy);
				float radius = worldSpaceZ * (kernelDiagonal / cameraNear);
				//gl_FragColor = vec4(fract(vec3(10000.0) + worldPosition / 100.0), 1.0);
				//return;
				// gl_FragColor = vec4(fract(10000.0 + worldPosition.x / 20.0), 0.0, 0.0, 1.0);
				// float dynamicMaxDistance = minDistance + radius - maxDistance;
				// End of Daniel Zhong's code

				vec2 noiseScale = vec2( resolution.x / 400.0, resolution.y / 400.0 );
				vec3 random = vec3( texture2D( tNoise, vUv * noiseScale ) );
				gl_FragColor = vec4( random, 1.0 );
				return;

				// compute matrix used to reorient a kernel vector

				vec3 tangent = normalize( random - worldNormal * dot( random, worldNormal ) );
				vec3 bitangent = cross( worldNormal, tangent );
				mat3 kernelMatrix = mat3( tangent, bitangent, worldNormal );

				// gl_FragColor = vec4(0.5 + 0.5 * worldNormal, 1.0);

				// gl_FragColor = vec4(0.5 + 0.5 * worldNormal, 1.0);
				// gl_FragColor = vec4(0.0, 0.5 + 0.5 * worldNormal.y, 0.0, 1.0);
				//gl_FragColor = vec4(vec3(length(worldNormal) / 2.0), 1.0);
				//return;

				//gl_FragColor = vec4(vec3(abs(viewZ) / 100.0), 1.0);
				//gl_FragColor = vec4(fract(100000.0 + worldPosition / 100.0), 1.0);
				//return;


				// gl_FragColor = vec4(random.x, random.y, random.z, 1.0);
								
				// vec4 ndc = cameraProjectionMatrix * cameraViewMatrix * vec4( worldPosition, 1.0 );
				// ndc /= ndc.w;
				// vec2 uv = ndc.xy * 0.5 + 0.5;
				// gl_FragColor = vec4(uv, 0., 1.0);
				// return;

				float AOScale = 3.0 * radius; // 30.0; // 30.0;
				float occlusion = 0.0;
				//gl_FragColor = vec4(vec3(radius / 100.0), 1.0);
				//return;
				for ( int i = 0; i < 0 + 1 * KERNEL_SIZE; i ++ ) {

					vec3 sampleVector = kernelMatrix * kernel[ i ]; // reorient sample vector in view space
					// vec3 samplePoint = worldPosition + ( sampleVector * kernelRadius ); // calculate sample point
					vec3 samplePoint = worldPosition + ( sampleVector * AOScale ); // calculate sample point
					// vec3 samplePoint = worldPosition + worldNormal * AOScale; // calculate sample point

					vec4 samplePointNDC = cameraProjectionMatrix * cameraViewMatrix * vec4( samplePoint, 1.0 ); // project point and calculate NDC
					samplePointNDC /= samplePointNDC.w;

					vec2 samplePointUv = samplePointNDC.xy * 0.5 + 0.5; // compute uv coordinates

					// gl_FragColor = vec4(vec3(10.0*length(samplePointUv - vUv)), 1.0);
					// gl_FragColor = vec4(0.5 + 10.0 * (samplePointUv - vUv), 0.0, 1.0);
					// gl_FragColor = vec4(vUv.x, vUv.y, 0.0, 1.0);

						// Old
						// float realDepth = getLinearDepth( samplePointUv ); // get linear depth from depth texture
						// float sampleDepth = viewZToOrthographicDepth( samplePoint.z, cameraNear, cameraFar ); // compute linear depth of the sample view Z value
						// float delta = sampleDepth - realDepth;
						// if ( delta  > minDistance && delta  < maxDistance ) {
						// 	occlusion += 1.0;
						// }

					float sampleDepth = getDepth( samplePointUv );
						/*
						// Daniel Zhong's code
						if (sampleDepth == 1.0) {
							// gl_FragColor = vec4(fract(vec3(10000.0) + worldPosition / 100.0), 1.0);
							//gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
							//return;
						}
						//gl_FragColor = vec4(vec3(sampleDepth) / 10.0, 1.0);
						*/
					
					float sampleViewZ = getViewZ( sampleDepth );
					vec3 sampleViewPosition = getViewPosition( samplePointUv, sampleDepth, sampleViewZ );
					vec3 sampleWorldPosition = getWorldPosition( sampleViewPosition );

					
					// vec3 sampleViewNormal = getViewNormal( samplePointUv );
					// vec3 sampleWorldNormal = getWorldNormal( sampleViewNormal );

					// vec3 dir = normalize(sampleWorldPosition - worldPosition);

					float worldDistance = length( sampleWorldPosition - worldPosition ) / AOScale;
					
					vec3 sampleDirection = normalize(sampleWorldPosition - worldPosition);
					float lightIntensity = clamp(dot(sampleDirection, normalize(worldNormal)), 0.0, 1.0);
					float distanceFadeout = clamp(1.0 - (worldDistance - 0.0) / 3.0, 0.0, 1.0);
					// occlusion += lightIntensity; //  * distanceFadeout;
					// occlusion += distanceFadeout;
					occlusion += lightIntensity * distanceFadeout;

					/*
						if (worldDistance < 10.0) {
							gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
						} else if (worldDistance < 30.0) {
							gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
						} else {
							gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
						}

						// float intensity = max(0.0, min(1.0, (worldDistance - 10.0) / 20.0));
						// gl_FragColor = vec4(vec3(1.0 - intensity), 1.0);

						float sampleWorldSpaceZ = dot(sampleWorldPosition - cameraPosition, -cameraInverseViewMatrix[2].xyz);
						float zDistance = abs(sampleWorldSpaceZ - worldSpaceZ);
						float relDistance = zDistance;

						
						float realDepth = dot(sampleWorldPosition - cameraPosition, -cameraInverseViewMatrix[2].xyz);
						sampleDepth = dot(samplePoint - cameraPosition, -cameraInverseViewMatrix[2].xyz);
						float delta = abs(sampleDepth - realDepth);
						if ( delta  > minDistance * radius && delta  < maxDistance * radius) {
							occlusion += 1.0;
						}

						// float normalDiff = max(0.0, dot(worldNormal, sampleWorldNormal));

						// if (relDistance > radius * 0.1) {
						// 	occlusion +=  max(0.0, min(1.0, 1.0 - (relDistance - radius * minDistance) / (radius * maxDistance - radius * minDistance)));
						// }

						// if ( relDistance > radius * minDistance && relDistance < radius * maxDistance ) {
						// 	occlusion += 1.0;
						// }
						*/

				}

				occlusion = clamp( 1.0 - 1.5 * occlusion / float( KERNEL_SIZE ), 0.0, 1.0 );
				gl_FragColor = vec4( vec3( occlusion ), 1.0 );
				// gl_FragColor = vec4( vec3( 1.0 - occlusion ), 1.0 );
				// gl_FragColor = vec4( worldNormal, 1.0 );
				// gl_FragColor = vec4( vec3( (worldSpaceZ / 1000.0) ), 1.0 );
			 // gl_FragColor = vec4( vec3( radius / maxDistance), 1.0 );
				// gl_FragColor = vec4( vec3( viewZ * 1000.0 + 0.5 ), 1.0 );
				// gl_FragColor = vec4( viewZ/5.0, 0.0, 0.0, 1.0 );
				
			}

		}`

};

const SSAODepthShader = {

	name: 'SSAODepthShader',

	defines: {
		'PERSPECTIVE_CAMERA': 1
	},

	uniforms: {

		'tDepth': { value: null },
		'cameraNear': { value: null },
		'cameraFar': { value: null },

	},

	vertexShader:

		`varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader:

		`uniform sampler2D tDepth;

		uniform float cameraNear;
		uniform float cameraFar;

		varying vec2 vUv;

		#include <packing>

		float getLinearDepth( const in vec2 screenPosition ) {

			#if PERSPECTIVE_CAMERA == 1

				float fragCoordZ = texture2D( tDepth, screenPosition ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );

			#else

				return texture2D( tDepth, screenPosition ).x;

			#endif

		}

		void main() {

			float depth = getLinearDepth( vUv );
			gl_FragColor = vec4( vec3( 1.0 - depth ), 1.0 );

		}`

};

const SSAOBlurShader = {

	name: 'SSAOBlurShader',

	uniforms: {

		'tDiffuse': { value: null },
		'resolution': { value: new Vector2() }

	},

	vertexShader:

		`varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader:

		`uniform sampler2D tDiffuse;

		uniform vec2 resolution;

		varying vec2 vUv;

		void main() {

			vec2 texelSize = ( 1.0 / resolution );
			float result = 0.0;

			for ( int i = - 2; i <= 2; i ++ ) {

				for ( int j = - 2; j <= 2; j ++ ) {

					vec2 offset = ( vec2( float( i ), float( j ) ) ) * texelSize;
					result += texture2D( tDiffuse, vUv + offset ).r;

				}

			}

			gl_FragColor = vec4( vec3( result / ( 5.0 * 5.0 ) ), 1.0 );

		}`

};

export { SSAOShader, SSAODepthShader, SSAOBlurShader };
