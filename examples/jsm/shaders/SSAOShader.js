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

		vec2 projectWorldPositionToScreenSpace(vec3 worldPos) {
			vec4 samplePointNDC = cameraProjectionMatrix * cameraViewMatrix * vec4( worldPos, 1.0 ); // project point and calculate NDC
			samplePointNDC /= samplePointNDC.w;
			return samplePointNDC.xy * 0.5 + 0.5; // compute uv coordinates
		}

		vec3 sampleDepthToWorldPosition(vec2 screenCoord, float sampleDepth) {
			float sampleViewZ = getViewZ( sampleDepth );
			vec3 sampleViewPosition = getViewPosition( screenCoord, sampleDepth, sampleViewZ );
			return getWorldPosition( sampleViewPosition );
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
								
				// Daniel Zhong's code
				vec3 worldPosition = getWorldPosition(viewPosition);
				vec3 worldNormal = getWorldNormal(viewNormal);
				
				float Sx = (float(kernelRadius) / 2.0) / resolution.x;
        		float Sy = (float(kernelRadius) / 2.0) / resolution.y;
		
				float worldSpaceZ = dot(worldPosition - cameraPosition, -cameraInverseViewMatrix[2].xyz);
				float kernelDiagonal = sqrt(Sx * Sx + Sy * Sy);
				float radius = worldSpaceZ * (kernelDiagonal / cameraNear);
				// float dynamicMaxDistance = minDistance + radius - maxDistance;

				vec3 random = 2.0 * (vec3( texture2D( tNoise, vUv * resolution / 1024.0 ) ) - 0.5);

				// compute matrix used to reorient a kernel vector

				vec3 tangent = normalize( random - worldNormal * dot( random, worldNormal ) );
				vec3 bitangent = cross( worldNormal, tangent );
				mat3 kernelMatrix = mat3( tangent, bitangent, worldNormal );

				float AOScale = 3.0 * radius;
				float occlusion = 0.0;

				for ( int i = 0; i < KERNEL_SIZE; i ++ ) {

					vec3 sampleVector = kernelMatrix * kernel[ i ]; // reorient sample vector in view space
					vec3 samplePoint = worldPosition + ( sampleVector * AOScale ); // calculate sample point

					if (length(sampleVector - samplePoint) > AOScale * 0.05) {

						vec2 samplePointUv = projectWorldPositionToScreenSpace(samplePoint);

						float sampleDepth = getDepth( samplePointUv );
					
						vec3 sampleWorldPosition = sampleDepthToWorldPosition(samplePointUv, sampleDepth);

						float worldDistance = length( sampleWorldPosition - worldPosition ) / AOScale;
						
						vec3 sampleDirection = normalize(sampleWorldPosition - worldPosition);
						float lightIntensity = clamp(dot(sampleDirection, normalize(worldNormal)), 0.0, 1.0);
						float distanceFadeout = clamp(1.0 - (worldDistance - 0.0) / 3.0, 0.0, 1.0);
						occlusion += lightIntensity * distanceFadeout / float( KERNEL_SIZE );
					}
				}

				occlusion = pow(1.0 - occlusion, 1.5);
				gl_FragColor = vec4( vec3( occlusion ), 1.0 );
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
