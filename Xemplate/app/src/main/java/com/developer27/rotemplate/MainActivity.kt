package com.developer27.rotemplate

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.preference.PreferenceManager
import com.developer27.rotemplate.camera.CameraHelper
import com.developer27.rotemplate.camera.RecorderHelper
import com.developer27.rotemplate.databinding.ActivityMainBinding
import com.developer27.rotemplate.videoprocessing.VideoProcessor
import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraHelper: CameraHelper
    private var videoProcessor: VideoProcessor? = null

    // RecorderHelper instance and flag to track recording state.
    private lateinit var recorderHelper: RecorderHelper
    private var isRecordingVideo = false
    private var isStartingVideo = false

    // Flags for tracking and frame processing.
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    private lateinit var cameraPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var recordingPermissionLauncher: ActivityResultLauncher<Array<String>>
    private val preferenceChangeListener =
        SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
            if (::cameraHelper.isInitialized && key in CAPTURE_SETTING_KEYS) {
                cameraHelper.updateCaptureSettings()
            }
        }

    companion object {
        private val CAPTURE_SETTING_KEYS = setOf(
            "shutter_speed",
            "manual_iso_enabled",
            "iso_value"
        )
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (cameraPermissionGranted()) {
                cameraHelper.openCamera()
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Prevent screen from turning off
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Install the splash screen (Android 12+)
        installSplashScreen()

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        // Initialize RecorderHelper for raw video recording.
        recorderHelper = RecorderHelper(this, cameraHelper, viewBinding)

        // Hide the processed frame view initially.
        viewBinding.processedFrameView.visibility = View.GONE

        // Open the URL when the title container is clicked.
        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        cameraPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
                if (granted) {
                    openCameraWhenReady()
                } else {
                    Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_SHORT).show()
                }
            }

        recordingPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
                if (permissions.values.all { it }) {
                    startVideoRecording()
                } else {
                    Toast.makeText(
                        this,
                        "Recording permission is required to save video.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

        // Set up the Start Tracking button.
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        // Set up the Capture Video button to trigger recording functions.
        viewBinding.startRecordingButton.setOnClickListener {
            if (isRecordingVideo || isStartingVideo) {
                stopVideoRecording()
            } else {
                val missingPermissions = recordingPermissions().filterNot(::permissionGranted)
                if (missingPermissions.isEmpty()) {
                    startVideoRecording()
                } else {
                    recordingPermissionLauncher.launch(missingPermissions.toTypedArray())
                }
            }
        }

        // Set up the Switch Camera, About, and Settings buttons.
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Load ML models.
        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener(preferenceChangeListener)
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE
        // No trace drawing is initialized.
    }

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false

        // Update UI and clear the processed frame view.
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        Toast.makeText(this, "Tracking stopped", Toast.LENGTH_LONG).show()
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, _) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    }
                }
                isProcessingFrame = false
            }
        }
    }

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(modelName)
            if (bestLoadedPath.isNotEmpty()) {
                try {
                    val modelBuffer = loadMappedFile(bestLoadedPath)
                    if (modelName == "YOLOv3_float32.tflite") {
                        videoProcessor?.loadInterpreter(modelBuffer) { error ->
                            if (error != null && !isDestroyed) {
                                Toast.makeText(
                                    this,
                                    "Error loading TFLite model: ${error.message}",
                                    Toast.LENGTH_LONG
                                ).show()
                            }
                        }
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.d("MainActivity", "TFLite Interpreter error", e)
                    }
                }
            } else {
                runOnUiThread {
                    Toast.makeText(this, "Failed to copy or load $modelName", Toast.LENGTH_SHORT).show()
                }
            }
        }.apply { name = "TFLiteLoader" }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        return file.inputStream().use { input ->
            input.channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            }
        }
    }

    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) {
                return outFile.absolutePath
            }
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4 * 1024)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                    }
                    output.flush()
                }
            }
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying asset $assetName: ${e.message}")
            ""
        }
    }

    private var isFrontCamera = false
    @SuppressLint("MissingPermission")
    private fun switchCamera() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        if (isRecordingVideo || isStartingVideo) {
            stopVideoRecording(restorePreview = false)
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        if (cameraPermissionGranted()) {
            cameraHelper.openCamera()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        openCameraWhenReady()
    }

    @SuppressLint("MissingPermission")
    private fun openCameraWhenReady() {
        if (viewBinding.viewFinder.isAvailable) {
            if (cameraPermissionGranted()) {
                cameraHelper.openCamera()
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        // If video recording is active, stop it.
        if (isRecordingVideo || isStartingVideo) {
            stopVideoRecording(restorePreview = false)
        }
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        sharedPreferences.unregisterOnSharedPreferenceChangeListener(preferenceChangeListener)
        videoProcessor?.close()
        videoProcessor = null
        super.onDestroy()
    }

    private fun startVideoRecording() {
        if (isRecordingVideo || isStartingVideo) return
        isStartingVideo = true
        viewBinding.startRecordingButton.isEnabled = false
        recorderHelper.startRecordingVideo { error ->
            isStartingVideo = false
            isRecordingVideo = error == null
            updateRecordingButton()
            if (error != null) {
                Toast.makeText(this, error, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun stopVideoRecording(restorePreview: Boolean = true) {
        recorderHelper.stopRecordingVideo(restorePreview)
        isStartingVideo = false
        isRecordingVideo = false
        updateRecordingButton()
    }

    private fun updateRecordingButton() {
        viewBinding.startRecordingButton.isEnabled = true
        viewBinding.startRecordingButton.text =
            if (isRecordingVideo) "Stop Video" else "Capture Video"
        viewBinding.startRecordingButton.backgroundTintList = ContextCompat.getColorStateList(
            this,
            if (isRecordingVideo) R.color.red else R.color.green
        )
    }

    private fun recordingPermissions(): Array<String> = buildList {
        add(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }.toTypedArray()

    private fun cameraPermissionGranted(): Boolean = permissionGranted(Manifest.permission.CAMERA)

    private fun permissionGranted(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
    }
}
