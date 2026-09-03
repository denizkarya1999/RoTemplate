package com.developer27.rotemplate.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import androidx.annotation.RequiresPermission
import com.developer27.rotemplate.MainActivity
import com.developer27.rotemplate.databinding.ActivityMainBinding

/**
 * CameraHelper is responsible for:
 *  - Opening & closing the camera
 *  - Switching front/back
 *  - Creating a preview
 *  - Handling zoom & shutter speed
 *  - Starting a background thread for camera operations
 *
 *  This version forces a specific AWB mode & color correction to avoid color tint on Pixel 4a.
 */
class CameraHelper(
    private val activity: MainActivity,
    private val viewBinding: ActivityMainBinding,
    private val sharedPreferences: SharedPreferences
) {
    // The Android Camera2 API
    val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    // Active camera device + capture session
    var cameraDevice: CameraDevice? = null
    var cameraCaptureSession: CameraCaptureSession? = null

    // Capture builder for preview (and record)
    var captureRequestBuilder: CaptureRequest.Builder? = null

    // Preview + video sizes
    var previewSize: Size? = null
    var videoSize: Size? = null

    // Sensor area for zoom
    var sensorArraySize: Rect? = null

    // Whether we are using the front camera
    var isFrontCamera = false

    // Thread for camera operations
    private var backgroundThread: HandlerThread? = null
    var backgroundHandler: Handler? = null
        private set

    // Zoom control
    private var zoomLevel = 1.0f
    private var maxZoom = 1.0f
    @Volatile
    private var isOpeningCamera = false
    @Volatile
    private var cameraOpenGeneration = 0

    /**
     * Callback for camera device events
     */
    private fun stateCallback(generation: Int) = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            if (!isOpeningCamera || generation != cameraOpenGeneration) {
                camera.close()
                return
            }
            isOpeningCamera = false
            cameraDevice = camera
            createCameraPreview()
        }

        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
            if (generation == cameraOpenGeneration) {
                isOpeningCamera = false
                if (cameraDevice === camera) cameraDevice = null
            }
        }

        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()
            if (generation != cameraOpenGeneration) return
            isOpeningCamera = false
            if (cameraDevice === camera) cameraDevice = null
            activity.runOnUiThread {
                Toast.makeText(activity, "Camera error ($error).", Toast.LENGTH_LONG).show()
            }
        }
    }

    // ------------------------------------------------------------------------
    // Background Thread Setup
    // ------------------------------------------------------------------------
    fun startBackgroundThread() {
        if (backgroundThread?.isAlive == true) return
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    // ------------------------------------------------------------------------
    // Open/Close Camera
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.CAMERA)
    fun openCamera() {
        if (cameraDevice != null || isOpeningCamera) return
        try {
            // Decide which camera (front/back)
            val cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // Grab the full sensor area for zoom
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            maxZoom = (characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)
                ?: 1.0f).coerceAtLeast(1.0f)
            zoomLevel = zoomLevel.coerceIn(1.0f, maxZoom)

            // Possible output sizes
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: return

            // Choose your preview/video sizes
            val previewChoices = map.getOutputSizes(SurfaceTexture::class.java)
            val videoChoices = map.getOutputSizes(MediaRecorder::class.java)
            if (previewChoices.isNullOrEmpty() || videoChoices.isNullOrEmpty()) {
                Toast.makeText(activity, "Camera does not expose compatible video sizes.", Toast.LENGTH_LONG).show()
                return
            }
            previewSize = chooseOptimalSize(previewChoices)
            videoSize = chooseOptimalSize(videoChoices)

            // Now open the selected camera
            val generation = cameraOpenGeneration + 1
            cameraOpenGeneration = generation
            isOpeningCamera = true
            cameraManager.openCamera(cameraId, stateCallback(generation), backgroundHandler)
        } catch (e: CameraAccessException) {
            isOpeningCamera = false
            e.printStackTrace()
        } catch (e: SecurityException) {
            isOpeningCamera = false
            e.printStackTrace()
            Toast.makeText(activity, "Camera permission needed.", Toast.LENGTH_SHORT).show()
        } catch (e: RuntimeException) {
            isOpeningCamera = false
            Log.e(TAG, "Unable to open camera", e)
            Toast.makeText(activity, "Unable to open the camera.", Toast.LENGTH_LONG).show()
        }
    }

    fun closeCamera() {
        cameraOpenGeneration += 1
        isOpeningCamera = false
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
        captureRequestBuilder = null
    }

    // ------------------------------------------------------------------------
    // Create Preview
    // ------------------------------------------------------------------------
    fun createCameraPreview() {
        try {
            val activeCamera = cameraDevice ?: return
            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            // Match the texture view size to the chosen preview size
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val previewSurface = Surface(texture)
            // Build a preview request
            captureRequestBuilder = activeCamera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            // Add the preview surface as a target
            captureRequestBuilder?.addTarget(previewSurface)

            applyCaptureSettings(submitRequest = false)

            // Now create the capture session
            cameraCaptureSession?.close()
            cameraCaptureSession = null
            activeCamera.createCaptureSession(
                listOf(previewSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice !== activeCamera) {
                            session.close()
                            return
                        }
                        // Save the session
                        cameraCaptureSession = session
                        updatePreview() // Start the preview
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        session.close()
                        Toast.makeText(
                            activity,
                            "Preview config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Unable to create camera preview", e)
        }
    }

    /**
     * Update the camera preview with latest builder settings
     */
    fun updatePreview() {
        if (cameraDevice == null) return
        try {
            val requestBuilder = captureRequestBuilder ?: return
            // Keep forcing color correction and AWB
            requestBuilder.set(
                CaptureRequest.CONTROL_AWB_MODE,
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            requestBuilder.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            cameraCaptureSession?.setRepeatingRequest(
                requestBuilder.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Unable to update camera preview", e)
        }
    }

    // ------------------------------------------------------------------------
    // Camera Selection (Front/Back)
    // ------------------------------------------------------------------------
    fun getCameraId(): String {
        for (id in cameraManager.cameraIdList) {
            val facing = cameraManager
                .getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (!isFrontCamera && facing == CameraCharacteristics.LENS_FACING_BACK) {
                return id
            } else if (isFrontCamera && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                return id
            }
        }
        // fallback if none matched
        return cameraManager.cameraIdList.firstOrNull()
            ?: throw IllegalStateException("No camera is available on this device.")
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        val targetWidth = 1280
        val targetHeight = 720

        // Try to find 1280x720 specifically
        val found720p = choices.find { it.width == targetWidth && it.height == targetHeight }
        if (found720p != null) {
            return found720p
        }
        // fallback to the smallest
        return choices.minByOrNull { it.width * it.height } ?: choices[0]
    }

    // ------------------------------------------------------------------------
    // Rolling shutter & exposure
    // ------------------------------------------------------------------------
    fun applyRollingShutter() {
        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val canManualExposure = capabilities?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true

        val shutterValueNs = CaptureSettings.exposureTimeNanos(
            sharedPreferences.getString(
                "shutter_speed",
                CaptureSettings.DEFAULT_SHUTTER_HZ.toString()
            )
        )

        val manualIsoEnabled = sharedPreferences.getBoolean("manual_iso_enabled", true)

        // Camera2 cannot choose ISO automatically while sensor exposure is fully manual.
        if (!canManualExposure || shutterValueNs == null || !manualIsoEnabled) {
            setAutoExposure()
            return
        }

        val exposureTimeRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)

        if (exposureTimeRange == null || isoRange == null) {
            setAutoExposure()
            return
        }

        val safeExposureNs = shutterValueNs.coerceIn(exposureTimeRange.lower, exposureTimeRange.upper)

        // Read ISO prefs
        val isoFromPrefs = CaptureSettings.normalizeIso(
            sharedPreferences.getString("iso_value", CaptureSettings.DEFAULT_ISO.toString())
        ) ?: CaptureSettings.DEFAULT_ISO
        val safeISO = isoFromPrefs.coerceIn(isoRange.lower, isoRange.upper)

        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_EXPOSURE_TIME, safeExposureNs)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_SENSITIVITY, safeISO)
    }

    private fun setAutoExposure() {
        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
    }

    /**
     * If user changes shutter speed in settings, we re-apply
     */
    fun updateCaptureSettings() {
        if (captureRequestBuilder == null || cameraCaptureSession == null) return
        try {
            applyRollingShutter()
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom(submitRequest = false)
            val request = captureRequestBuilder?.build() ?: return
            cameraCaptureSession?.setRepeatingRequest(
                request,
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Unable to apply camera settings", e)
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "Camera rejected updated settings", e)
        }
    }

    fun applyCaptureSettings(submitRequest: Boolean = true) {
        applyRollingShutter()
        applyFlashIfEnabled()
        applyLightingMode()
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        captureRequestBuilder?.set(
            CaptureRequest.COLOR_CORRECTION_MODE,
            CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
        )
        applyZoom(submitRequest = false)
        if (submitRequest) updatePreview()
    }

    // ------------------------------------------------------------------------
    // Flash & Lighting
    // ------------------------------------------------------------------------
    fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        val hasFlash = runCatching {
            cameraManager.getCameraCharacteristics(getCameraId())
                .get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
        }.getOrDefault(false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled && hasFlash) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
    }

    fun applyLightingMode() {
        // Only apply AE compensation if AE is ON
        val aeMode = captureRequestBuilder?.get(CaptureRequest.CONTROL_AE_MODE)
        if (aeMode == CameraMetadata.CONTROL_AE_MODE_ON) {
            val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
            val cameraId = getCameraId()
            val compensationRange = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

            val exposureComp = when (lightingMode) {
                "low_light" -> compensationRange?.lower ?: 0
                "high_light" -> compensationRange?.upper ?: 0
                else -> 0
            }
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                exposureComp
            )
        }
    }

    // ------------------------------------------------------------------------
    // Zoom
    // ------------------------------------------------------------------------
    fun setupZoomControls() {
        val zoomHandler = Handler(activity.mainLooper)
        var zoomInRunnable: Runnable? = null
        var zoomOutRunnable: Runnable? = null

        // Repetitive zoom in on long-press
        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRunnable = object : Runnable {
                        override fun run() {
                            zoomIn()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomInRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomInRunnable?.let(zoomHandler::removeCallbacks)
                    true
                }
                else -> false
            }
        }

        // Repetitive zoom out on long-press
        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRunnable = object : Runnable {
                        override fun run() {
                            zoomOut()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomOutRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomOutRunnable?.let(zoomHandler::removeCallbacks)
                    true
                }
                else -> false
            }
        }
    }

    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel += 0.1f
            applyZoom()
        }
    }

    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel -= 0.1f
            applyZoom()
        }
    }

    /**
     * Applies digital zoom by setting the SCALER_CROP_REGION
     */
    fun applyZoom(submitRequest: Boolean = true) {
        val sensorRect = sensorArraySize ?: return
        val requestBuilder = captureRequestBuilder ?: return
        zoomLevel = zoomLevel.coerceIn(1.0f, maxZoom)
        val ratio = 1 / zoomLevel
        val croppedWidth = sensorRect.width() * ratio
        val croppedHeight = sensorRect.height() * ratio

        val left = sensorRect.left + ((sensorRect.width() - croppedWidth) / 2).toInt()
        val top = sensorRect.top + ((sensorRect.height() - croppedHeight) / 2).toInt()
        val right = (left + croppedWidth).toInt()
        val bottom = (top + croppedHeight).toInt()

        val zoomRect = Rect(left, top, right, bottom)
        requestBuilder.set(CaptureRequest.SCALER_CROP_REGION, zoomRect)

        if (!submitRequest) return
        try {
            cameraCaptureSession?.setRepeatingRequest(
                requestBuilder.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Unable to update zoom", e)
        }
    }

    private companion object {
        const val TAG = "CameraHelper"
    }
}
