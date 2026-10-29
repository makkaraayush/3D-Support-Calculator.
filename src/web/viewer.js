/**
 * Three.js 3D Viewport Controller for 3D Support Calculator.
 * Handles rendering the Bambu Lab A1 build plate, model geometry,
 * overhang angle color heatmaps, and generated tree support structures.
 */

class SupportViewer3D {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // Objects
    this.bedGroup = null;
    this.modelMesh = null;
    this.supportMesh = null;

    // Visibility flags
    this.showModel = true;
    this.showHeatmap = true;
    this.showSupport = true;
    this.showBed = true;

    // Stored geometry data
    this.currentModelData = null;

    this.init();
  }

  init() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060910);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 2000);
    this.camera.position.set(180, -220, 160);
    this.camera.up.set(0, 0, 1); // Z is UP in 3D printing

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.target.set(0, 0, 20);
    }

    // 5. Lighting
    this.setupLighting();

    // 6. Bambu Lab A1 Build Bed (256 x 256 mm)
    this.setupBambuBed();

    // 7. Event Listeners
    window.addEventListener('resize', () => this.onResize());

    // 8. Animation Loop
    this.animate();
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(200, 250, 400);
    dirLight1.castShadow = true;
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x00d8f6, 0.35); // subtle cool rim light
    dirLight2.position.set(-200, -200, -100);
    this.scene.add(dirLight2);
  }

  setupBambuBed() {
    this.bedGroup = new THREE.Group();

    // 256 x 256 mm build plate
    const bedSize = 256;
    const gridHelper = new THREE.GridHelper(bedSize, 32, 0x00d8f6, 0x1f293d);
    gridHelper.rotation.x = Math.PI / 2; // Rotate so grid lies in XY plane at Z=0
    this.bedGroup.add(gridHelper);

    // Bed plate edge border
    const borderGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-128, -128, 0),
      new THREE.Vector3(128, -128, 0),
      new THREE.Vector3(128, 128, 0),
      new THREE.Vector3(-128, 128, 0),
      new THREE.Vector3(-128, -128, 0),
    ]);
    const borderMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 2 });
    const borderLine = new THREE.Line(borderGeo, borderMat);
    this.bedGroup.add(borderLine);

    this.scene.add(this.bedGroup);
  }

  onResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.controls) this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  loadGeometry(data) {
    this.currentModelData = data;

    // 1. Clear old model mesh
    if (this.modelMesh) {
      this.scene.remove(this.modelMesh);
      if (this.modelMesh.geometry) this.modelMesh.geometry.dispose();
      this.modelMesh = null;
    }

    if (!data.has_model || !data.vertices || data.vertices.length === 0) {
      return;
    }

    // 2. Build Model BufferGeometry
    const vArr = new Float32Array(data.vertices);
    const fArr = new Uint32Array(data.faces);

    // Center geometry around origin XY on the bed Z=0
    // Centering offsets
    const bMin = data.bounds[0];
    const bMax = data.bounds[1];
    const cX = (bMin[0] + bMax[0]) * 0.5;
    const cY = (bMin[1] + bMax[1]) * 0.5;

    const centeredVerts = new Float32Array(vArr.length);
    for (let i = 0; i < vArr.length; i += 3) {
      centeredVerts[i] = vArr[i] - cX;
      centeredVerts[i + 1] = vArr[i + 1] - cY;
      centeredVerts[i + 2] = vArr[i + 2];
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(centeredVerts, 3));
    geometry.setIndex(new THREE.BufferAttribute(fArr, 1));
    geometry.computeVertexNormals();

    // 3. Compute per-face or per-vertex colors for overhang heatmap
    const colors = new Float32Array(centeredVerts.length);
    const angles = data.face_angles || [];

    // Map overhang angle to color:
    // <= 55 deg: cool silver-slate (#8fa0b5)
    // 55 - 70 deg: warning amber (#f59e0b)
    // > 70 deg: danger crimson (#ef4444)
    for (let i = 0; i < centeredVerts.length; i += 3) {
      // Default sleek slate
      colors[i] = 0.58;
      colors[i + 1] = 0.65;
      colors[i + 2] = 0.74;
    }

    if (this.showHeatmap && angles.length > 0) {
      for (let fi = 0; fi < fArr.length; fi += 3) {
        const faceIdx = Math.floor(fi / 3);
        const ang = angles[faceIdx] || 0.0;

        let r = 0.58, g = 0.65, b = 0.74;
        if (ang >= 70.0) {
          r = 0.94; g = 0.27; b = 0.27; // severe overhang red
        } else if (ang >= 55.0) {
          r = 0.96; g = 0.62; b = 0.04; // warning amber
        }

        // Assign to 3 face vertices
        for (let j = 0; j < 3; j++) {
          const vIdx = fArr[fi + j] * 3;
          colors[vIdx] = r;
          colors[vIdx + 1] = g;
          colors[vIdx + 2] = b;
        }
      }
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.35,
      metalness: 0.1,
      side: THREE.DoubleSide,
    });

    this.modelMesh = new THREE.Mesh(geometry, material);
    this.modelMesh.visible = this.showModel;
    this.scene.add(this.modelMesh);

    // 4. Update Support Mesh
    this.updateSupportGeometry(data.support, cX, cY);

    // 5. Fit camera view
    this.fitCameraToMesh(data.bounds);
  }

  updateSupportGeometry(supportData, cX = 0, cY = 0) {
    if (this.supportMesh) {
      this.scene.remove(this.supportMesh);
      if (this.supportMesh.geometry) this.supportMesh.geometry.dispose();
      this.supportMesh = null;
    }

    if (!supportData || !supportData.has_support || supportData.vertices.length === 0) {
      return;
    }

    const sVerts = new Float32Array(supportData.vertices);
    const sFaces = new Uint32Array(supportData.faces);

    const centeredSVerts = new Float32Array(sVerts.length);
    for (let i = 0; i < sVerts.length; i += 3) {
      centeredSVerts[i] = sVerts[i] - cX;
      centeredSVerts[i + 1] = sVerts[i + 1] - cY;
      centeredSVerts[i + 2] = sVerts[i + 2];
    }

    const sGeo = new THREE.BufferGeometry();
    sGeo.setAttribute('position', new THREE.BufferAttribute(centeredSVerts, 3));
    sGeo.setIndex(new THREE.BufferAttribute(sFaces, 1));
    sGeo.computeVertexNormals();

    const sMat = new THREE.MeshStandardMaterial({
      color: 0x00d8f6, // luminous electric cyan for AI support branches
      roughness: 0.25,
      metalness: 0.2,
      emissive: 0x00384d,
      side: THREE.DoubleSide,
    });

    this.supportMesh = new THREE.Mesh(sGeo, sMat);
    this.supportMesh.visible = this.showSupport;
    this.scene.add(this.supportMesh);
  }

  fitCameraToMesh(bounds) {
    if (!bounds || !this.controls) return;
    const sizeX = bounds[1][0] - bounds[0][0];
    const sizeY = bounds[1][1] - bounds[0][1];
    const sizeZ = bounds[1][2] - bounds[0][2];
    const maxDim = Math.max(sizeX, sizeY, sizeZ, 30);

    const dist = maxDim * 2.8;
    this.camera.position.set(dist * 0.7, -dist * 0.9, dist * 0.8);
    this.controls.target.set(0, 0, sizeZ * 0.5);
    this.controls.update();
  }

  toggleModel() {
    this.showModel = !this.showModel;
    if (this.modelMesh) this.modelMesh.visible = this.showModel;
    return this.showModel;
  }

  toggleHeatmap() {
    this.showHeatmap = !this.showHeatmap;
    if (this.currentModelData) {
      this.loadGeometry(this.currentModelData);
    }
    return this.showHeatmap;
  }

  toggleSupport() {
    this.showSupport = !this.showSupport;
    if (this.supportMesh) this.supportMesh.visible = this.showSupport;
    return this.showSupport;
  }

  toggleBed() {
    this.showBed = !this.showBed;
    if (this.bedGroup) this.bedGroup.visible = this.showBed;
    return this.showBed;
  }

  resetView() {
    if (this.currentModelData && this.currentModelData.bounds) {
      this.fitCameraToMesh(this.currentModelData.bounds);
    } else {
      this.camera.position.set(180, -220, 160);
      if (this.controls) {
        this.controls.target.set(0, 0, 20);
        this.controls.update();
      }
    }
  }
}

window.SupportViewer3D = SupportViewer3D;
