import {
	AddEquation,
	Color,
	CustomBlending,
	DataTexture,
	DepthTexture,
	DstAlphaFactor,
	DstColorFactor,
	FloatType,
	HalfFloatType,
	MathUtils,
	MeshNormalMaterial,
	NearestFilter,
	NoBlending,
	RedFormat,
	DepthStencilFormat,
	UnsignedInt248Type,
	RepeatWrapping,
	ShaderMaterial,
	UniformsUtils,
	Vector3,
	WebGLRenderTarget,
	ZeroFactor,
	TextureLoader
} from 'three';
import { Pass, FullScreenQuad } from './Pass.js';
import { SimplexNoise } from '../math/SimplexNoise.js';
import { SSAOShader } from '../shaders/SSAOShader.js';
import { SSAOBlurShader } from '../shaders/SSAOShader.js';
import { SSAOShaderOld } from '../shaders/SSAOShaderOld.js';
import { SSAOBlurShaderOld } from '../shaders/SSAOShaderOld.js';
import { SSAODepthShader } from '../shaders/SSAOShader.js';
import { CopyShader } from '../shaders/CopyShader.js';
import { ImprovedNoise } from '../math/ImprovedNoise.js';

class SSAOPass extends Pass {

	constructor( scene, camera, width, height, kernelSize = 32 ) {

		super();

		this.width = ( width !== undefined ) ? width : 512;
		this.height = ( height !== undefined ) ? height : 512;

		this.clear = true;
		this.needsSwap = false;

		this.camera = camera;
		this.scene = scene;

		this.kernelRadius = 8;
		this.kernel = [];
		this.noiseTexture = null;
		this.output = 0;

		this.minDistance = 0.005;
		this.maxDistance = 0.1;
		this.aoPower = 1.5;
        this.blurScale = 1.0; 
		this.blurSampleCount = 3;
		this._visibilityCache = new Map();

		const storedKernelSize = localStorage.getItem('kernelSize');
        this.kernelSize = storedKernelSize ? parseInt(storedKernelSize, 10) : kernelSize;

		

		this.advancedSSAO = new ShaderMaterial({
			defines: Object.assign({}, SSAOShader.defines),
			uniforms: UniformsUtils.clone(SSAOShader.uniforms),
			vertexShader: SSAOShader.vertexShader,
			fragmentShader: SSAOShader.fragmentShader,
			blending: NoBlending
		} );

		this.oldSSAO = new ShaderMaterial({
			defines: Object.assign({}, SSAOShaderOld.defines),
			uniforms: UniformsUtils.clone(SSAOShaderOld.uniforms),
			vertexShader: SSAOShaderOld.vertexShader,
			fragmentShader: SSAOShaderOld.fragmentShader,
			blending: NoBlending
		} );

		this.generateSampleKernel( kernelSize );
		this.loadBlueNoiseTexture();
		this.generateRandomKernelRotations();

		// depth texture

		const depthTexture = new DepthTexture();
		depthTexture.format = DepthStencilFormat;
		depthTexture.type = UnsignedInt248Type;

		// normal render target with depth buffer

		this.normalRenderTarget = new WebGLRenderTarget( this.width, this.height, {
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			type: HalfFloatType,
			depthTexture: depthTexture
		} );

		// ssao render target

		this.ssaoRenderTarget = new WebGLRenderTarget( this.width, this.height, { type: HalfFloatType } );

		this.blurRenderTarget = this.ssaoRenderTarget.clone();

		// ssao material

		const savedShaderType = localStorage.getItem('selectedShader') || 'Advanced'; // Default to 'Advanced'

        // Apply the correct shader based on the stored value
        if (savedShaderType === 'Advanced') {
            this.ssaoMaterial = this.advancedSSAO;
        } else if (savedShaderType === 'SSAO Old') {
            this.ssaoMaterial = this.oldSSAO;
        } else {
            console.warn('Unknown SSAO shader type: ' + savedShaderType);
            this.ssaoMaterial = this.advancedSSAO;  // Fallback to 'Advanced'
        }

		this.ssaoMaterial.defines[ 'KERNEL_SIZE' ] = kernelSize;

		this.ssaoMaterial.uniforms[ 'tNormal' ].value = this.normalRenderTarget.texture;
		this.ssaoMaterial.uniforms[ 'tDepth' ].value = this.normalRenderTarget.depthTexture;
		// this.ssaoMaterial.uniforms[ 'tNoise' ].value = this.noiseTexture;
		this.ssaoMaterial.uniforms[ 'kernel' ].value = this.kernel;
		this.ssaoMaterial.uniforms[ 'cameraNear' ].value = this.camera.near;
		this.ssaoMaterial.uniforms[ 'cameraFar' ].value = this.camera.far;
		this.ssaoMaterial.uniforms[ 'resolution' ].value.set( this.width, this.height );
		this.ssaoMaterial.uniforms[ 'cameraProjectionMatrix' ].value.copy( this.camera.projectionMatrix );
		this.ssaoMaterial.uniforms[ 'cameraInverseProjectionMatrix' ].value.copy( this.camera.projectionMatrixInverse );
		this.ssaoMaterial.uniforms[ 'cameraInverseViewMatrix' ].value.copy( this.camera.matrixWorld );
		this.ssaoMaterial.uniforms[ 'cameraViewMatrix' ].value.copy( this.camera.matrixWorldInverse );
		this.ssaoMaterial.uniforms[ 'cameraPosition' ] = { value: new Vector3() };
		if (this.ssaoMaterial.uniforms['aoPower'] !== undefined) {
			this.ssaoMaterial.uniforms['aoPower'].value = this.aoPower;
		}
		
		if (this.ssaoMaterial.uniforms['kernelSize'] !== undefined) {
			this.ssaoMaterial.uniforms['kernelSize'].value = this.kernelSize;
		}

		// normal material

		this.normalMaterial = new MeshNormalMaterial();
		this.normalMaterial.blending = NoBlending;

		// blur material

		this.blurMaterialNew = new ShaderMaterial({
            defines: Object.assign({}, SSAOBlurShader.defines),
            uniforms: UniformsUtils.clone(SSAOBlurShader.uniforms),
            vertexShader: SSAOBlurShader.vertexShader,
            fragmentShader: SSAOBlurShader.fragmentShader
        });

        this.blurMaterialOld = new ShaderMaterial({
            defines: Object.assign({}, SSAOBlurShaderOld.defines),
            uniforms: UniformsUtils.clone(SSAOBlurShaderOld.uniforms),
            vertexShader: SSAOBlurShaderOld.vertexShader,
            fragmentShader: SSAOBlurShaderOld.fragmentShader
        });

		const savedBlurShaderType = localStorage.getItem('blurShaderType') || 'Blur New';
		if (savedBlurShaderType === 'Blur New') {
			this.blurMaterial = this.blurMaterialNew;
		} else if (savedBlurShaderType === 'Blur Old') {
			this.blurMaterial = this.blurMaterialOld;
		} else {
			this.blurMaterial = this.blurMaterialNew;
		}

		this.blurMaterial.uniforms[ 'tDiffuse' ].value = this.ssaoRenderTarget.texture;
		this.blurMaterial.uniforms[ 'resolution' ].value.set( this.width, this.height );
		
		if (this.blurMaterial.uniforms['tNormal'] !== undefined) {
			this.blurMaterial.uniforms['tNormal'].value = this.normalRenderTarget.texture;
		}
		if (this.blurMaterial.uniforms['blurScale'] !== undefined) {
			this.blurMaterial.uniforms['blurScale'].value = this.blurScale;
		}
		

		// material for rendering the depth

		this.depthRenderMaterial = new ShaderMaterial( {
			defines: Object.assign( {}, SSAODepthShader.defines ),
			uniforms: UniformsUtils.clone( SSAODepthShader.uniforms ),
			vertexShader: SSAODepthShader.vertexShader,
			fragmentShader: SSAODepthShader.fragmentShader,
			blending: NoBlending
		} );
		this.depthRenderMaterial.uniforms[ 'tDepth' ].value = this.normalRenderTarget.depthTexture;
		this.depthRenderMaterial.uniforms[ 'cameraNear' ].value = this.camera.near;
		this.depthRenderMaterial.uniforms[ 'cameraFar' ].value = this.camera.far;

		// material for rendering the content of a render target

		this.copyMaterial = new ShaderMaterial( {
			uniforms: UniformsUtils.clone( CopyShader.uniforms ),
			vertexShader: CopyShader.vertexShader,
			fragmentShader: CopyShader.fragmentShader,
			transparent: true,
			depthTest: false,
			depthWrite: false,
			blendSrc: DstColorFactor,
			blendDst: ZeroFactor,
			blendEquation: AddEquation,
			blendSrcAlpha: DstAlphaFactor,
			blendDstAlpha: ZeroFactor,
			blendEquationAlpha: AddEquation
		} );

		this.fsQuad = new FullScreenQuad( null );

		this.originalClearColor = new Color();

	}

	setKernelSize(size) {
        this.kernelSize = size;
        localStorage.setItem('kernelSize', size);  // Save the value in localStorage

        // Regenerate the kernel and apply it dynamically without reloading the page
        this.generateSampleKernel(size);  // Regenerate the kernel with the new size
        if (this.ssaoMaterial.uniforms['kernel'] !== undefined) {
            this.ssaoMaterial.uniforms['kernel'].value = this.kernel;
        }
        if (this.ssaoMaterial.defines['KERNEL_SIZE'] !== undefined) {
            this.ssaoMaterial.defines['KERNEL_SIZE'] = size;
        }
        this.ssaoMaterial.needsUpdate = true;  // Force the material to update
    }

	toggleBlurShader(type) {
        localStorage.setItem('selectedBlurShader', type);

        if (type === 'Blur New') {
            this.blurMaterial = this.blurMaterialNew;
        } else if (type === 'Blur Old') {
            this.blurMaterial = this.blurMaterialOld;
        }

        window.location.reload();
    }

	toggleShader(type) {
		localStorage.setItem('selectedShader', type);

		if (type === 'Advanced') {
			this.minDistance = 0.0;
			this.maxDistance = 3.0;
		} else if (type === 'SSAO Old') {
			this.minDistance = 0.005;
			this.maxDistance = 0.1;
		}

		window.location.reload();
	}
	setAOPower(power) {
		this.aoPower = power;
	
		if (this.ssaoMaterial.uniforms['aoPower'] !== undefined) {
			this.ssaoMaterial.uniforms['aoPower'].value = power;
		}
	}
	
	setBlurScale(scale) {
		this.blurScale = scale;
	
		if (this.blurMaterial.uniforms['blurScale'] !== undefined) {
			this.blurMaterial.uniforms['blurScale'].value = scale;
		}
	}
	
	setBlurSampleCount(sampleCount) {
        if (this.blurMaterial.uniforms['blurSampleCount'] !== undefined) {
            this.blurMaterial.uniforms['blurSampleCount'].value = sampleCount;
        }
    }

	dispose() {

		// dispose render targets

		this.normalRenderTarget.dispose();
		this.ssaoRenderTarget.dispose();
		this.blurRenderTarget.dispose();

		// dispose materials

		this.normalMaterial.dispose();
		this.blurMaterial.dispose();
		this.copyMaterial.dispose();
		this.depthRenderMaterial.dispose();

		// dipsose full screen quad

		this.fsQuad.dispose();

	}

	loadBlueNoiseTexture() {
		const loader = new TextureLoader();
	
		// Load your blue noise texture
		loader.load('./jsm/postprocessing/blueNoise.png', (texture) => {
		  // Set texture properties
		  texture.wrapS = RepeatWrapping;
		  texture.wrapT = RepeatWrapping;
		  texture.magFilter = NearestFilter;
		  texture.minFilter = NearestFilter;
		  this.noiseTexture = texture; // Assign the loaded texture to SSAO shader uniform
		  this.ssaoMaterial.uniforms['tNoise'].value = this.noiseTexture;
		  if (this.blurMaterial.uniforms['tNoise'] !== undefined) {
			this.blurMaterial.uniforms['tNoise'].value = this.noiseTexture;
		}
		
		});
	  }

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {
		// render normals and depth (honor only meshes, points and lines do not contribute to SSAO)

		this.camera.updateMatrixWorld( true );
   		this.camera.updateProjectionMatrix();

		// Rebind the camera matrices to the SSAO shader uniforms
		this.ssaoMaterial.uniforms[ 'cameraProjectionMatrix' ].value.copy( this.camera.projectionMatrix );
		this.ssaoMaterial.uniforms[ 'cameraInverseProjectionMatrix' ].value.copy( this.camera.projectionMatrixInverse );
		this.ssaoMaterial.uniforms[ 'cameraInverseViewMatrix' ].value.copy( this.camera.matrixWorld );
		this.ssaoMaterial.uniforms[ 'cameraViewMatrix' ].value.copy( this.camera.matrixWorldInverse );
		

		this.ssaoMaterial.uniforms[ 'resolution' ].value.set( this.width, this.height );
		// render normals and depth (honor only meshes, points, and lines do not contribute to SSAO)
		this.overrideVisibility();
		this.renderOverride( renderer, this.normalMaterial, this.normalRenderTarget, 0x7777ff, 1.0 );
		this.restoreVisibility();

		// render SSAO

		this.ssaoMaterial.uniforms[ 'kernelRadius' ].value = this.kernelRadius;
		this.ssaoMaterial.uniforms[ 'minDistance' ].value = this.minDistance;
		this.ssaoMaterial.uniforms[ 'maxDistance' ].value = this.maxDistance;
		if (this.ssaoMaterial.uniforms['tDiffuse']) {
			this.ssaoMaterial.uniforms['tDiffuse'].value = readBuffer.texture;
		}
		this.ssaoMaterial.uniforms[ 'cameraPosition' ].value.copy( this.camera.position );
		this.renderPass( renderer, this.ssaoMaterial, this.ssaoRenderTarget );

		// render blur

		this.renderPass( renderer, this.blurMaterial, this.blurRenderTarget );

		// output result to screen

		switch ( this.output ) {

			case SSAOPass.OUTPUT.SSAO:

				this.copyMaterial.uniforms[ 'tDiffuse' ].value = this.ssaoRenderTarget.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : readBuffer );

				break;

			case SSAOPass.OUTPUT.Blur:

				this.copyMaterial.uniforms[ 'tDiffuse' ].value = this.blurRenderTarget.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : readBuffer );

				break;

			case SSAOPass.OUTPUT.Depth:

				this.renderPass( renderer, this.depthRenderMaterial, this.renderToScreen ? null : readBuffer );

				break;

			case SSAOPass.OUTPUT.Normal:

				this.copyMaterial.uniforms[ 'tDiffuse' ].value = this.normalRenderTarget.texture;
				this.copyMaterial.blending = NoBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : readBuffer );

				break;

			case SSAOPass.OUTPUT.Default:

				this.copyMaterial.uniforms[ 'tDiffuse' ].value = this.blurRenderTarget.texture;
				this.copyMaterial.blending = CustomBlending;
				this.renderPass( renderer, this.copyMaterial, this.renderToScreen ? null : readBuffer );

				break;

			default:
				console.warn( 'THREE.SSAOPass: Unknown output type.' );

		}

	}

	renderPass( renderer, passMaterial, renderTarget, clearColor, clearAlpha ) {

		// save original state
		renderer.getClearColor( this.originalClearColor );
		const originalClearAlpha = renderer.getClearAlpha();
		const originalAutoClear = renderer.autoClear;

		renderer.setRenderTarget( renderTarget );

		// setup pass state
		renderer.autoClear = false;
		if ( ( clearColor !== undefined ) && ( clearColor !== null ) ) {

			renderer.setClearColor( clearColor );
			renderer.setClearAlpha( clearAlpha || 0.0 );
			renderer.clear();

		}

		this.fsQuad.material = passMaterial;
		this.fsQuad.render( renderer );

		// restore original state
		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( this.originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );

	}

	renderOverride( renderer, overrideMaterial, renderTarget, clearColor, clearAlpha ) {

		renderer.getClearColor( this.originalClearColor );
		const originalClearAlpha = renderer.getClearAlpha();
		const originalAutoClear = renderer.autoClear;

		renderer.setRenderTarget( renderTarget );
		renderer.autoClear = false;

		clearColor = overrideMaterial.clearColor || clearColor;
		clearAlpha = overrideMaterial.clearAlpha || clearAlpha;

		if ( ( clearColor !== undefined ) && ( clearColor !== null ) ) {

			renderer.setClearColor( clearColor );
			renderer.setClearAlpha( clearAlpha || 0.0 );
			renderer.clear();

		}

		this.scene.overrideMaterial = overrideMaterial;
		renderer.render( this.scene, this.camera );
		this.scene.overrideMaterial = null;

		// restore original state

		renderer.autoClear = originalAutoClear;
		renderer.setClearColor( this.originalClearColor );
		renderer.setClearAlpha( originalClearAlpha );

	}

	setSize( width, height ) {

		this.width = width;
		this.height = height;

		this.ssaoRenderTarget.setSize( width, height );
		this.normalRenderTarget.setSize( width, height );
		this.blurRenderTarget.setSize( width, height );

		this.ssaoMaterial.uniforms[ 'resolution' ].value.set( width, height );
		this.ssaoMaterial.uniforms[ 'cameraProjectionMatrix' ].value.copy( this.camera.projectionMatrix );
		this.ssaoMaterial.uniforms[ 'cameraInverseProjectionMatrix' ].value.copy( this.camera.projectionMatrixInverse );
		this.ssaoMaterial.uniforms[ 'cameraInverseViewMatrix' ].value.copy( this.camera.matrixWorld );
		this.ssaoMaterial.uniforms[ 'cameraViewMatrix' ].value.copy( this.camera.matrixWorldInverse );

		this.blurMaterial.uniforms[ 'resolution' ].value.set( width, height );

	}

	generateSampleKernel( kernelSize ) {

		const kernel = this.kernel;

		for ( let i = 0; i < kernelSize; i ++ ) {

			const sample = new Vector3();
			// sample.x = ( Math.random() * 2 ) - 1;
			// sample.y = ( Math.random() * 2 ) - 1;
			// sample.z = 0.05 + Math.random(); // We don't want Z to be 0
			
			// sample.normalize();
			
			// // Importance sampling
			// sample.x *= Math.abs(sample.x);
			// sample.y *= Math.abs(sample.y);

			// sample.normalize();

			const u1 = Math.random();
			const u2 = Math.random();

			const r = Math.sqrt(u1);
			const theta = 2 * Math.PI * u2;

			sample.x = r * Math.cos(theta);
			sample.y = r * Math.sin(theta);
			sample.z = Math.sqrt(1 - u1);

			// Scale must not be 0
			let scale = (i + 1) / kernelSize;
			scale = Math.pow( scale, 2 );
			sample.multiplyScalar( scale );

			kernel.push( sample );

		}

	}

	generateRandomKernelRotations() {

		const width = 4, height = 4;

		const noiseGenerator = new ImprovedNoise();

		const size = width * height;
		const data = new Float32Array( size );

		for ( let i = 0; i < size; i ++ ) {

			const x = Math.random() * 100; // Larger scale for better texture generation
			const y = Math.random() * 100;
			const z = 0;

			// Generate Perlin noise using ImprovedNoise for each point
			data[ i ] = noiseGenerator.noise( x, y, z );

		}

		// Creating the noise texture
		this.noiseTexture = new DataTexture( data, width, height, RedFormat, FloatType );
		this.noiseTexture.wrapS = RepeatWrapping;
		this.noiseTexture.wrapT = RepeatWrapping;
		this.noiseTexture.needsUpdate = true;

	}

	overrideVisibility() {

		const scene = this.scene;
		const cache = this._visibilityCache;

		scene.traverse( function ( object ) {

			cache.set( object, object.visible );

			if ( object.isPoints || object.isLine ) object.visible = false;

		} );

	}

	restoreVisibility() {

		const scene = this.scene;
		const cache = this._visibilityCache;

		scene.traverse( function ( object ) {

			const visible = cache.get( object );
			object.visible = visible;

		} );

		cache.clear();

	}

}

SSAOPass.OUTPUT = {
	'Default': 0,
	'SSAO': 1,
	'Blur': 2,
	'Depth': 3,
	'Normal': 4
};

export { SSAOPass };
