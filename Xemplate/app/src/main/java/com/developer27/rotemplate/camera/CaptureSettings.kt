package com.developer27.rotemplate.camera

internal object CaptureSettings {
    const val DEFAULT_SHUTTER_HZ = 6000
    const val DEFAULT_ISO = 1100
    const val MIN_ISO = 50
    const val MAX_ISO = 25600

    fun exposureTimeNanos(rawShutterHz: String?): Long? {
        val shutterHz = rawShutterHz?.toLongOrNull() ?: DEFAULT_SHUTTER_HZ.toLong()
        return if (shutterHz > 0L) 1_000_000_000L / shutterHz else null
    }

    fun normalizeIso(rawIso: String?): Int? {
        return rawIso?.trim()?.toIntOrNull()?.coerceIn(MIN_ISO, MAX_ISO)
    }
}
