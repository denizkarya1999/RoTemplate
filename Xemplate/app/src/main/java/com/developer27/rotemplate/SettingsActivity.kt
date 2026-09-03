package com.developer27.rotemplate

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference
import com.developer27.rotemplate.camera.CaptureSettings

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)

        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)
            val manualIsoPref = findPreference<SwitchPreference>("manual_iso_enabled")

            val isoPref = findPreference<EditTextPreference>("iso_value")
            isoPref?.isEnabled = manualIsoPref?.isChecked != false
            isoPref?.setOnBindEditTextListener { edit ->
                edit.hint = "${CaptureSettings.MIN_ISO}–${CaptureSettings.MAX_ISO}"
            }

            manualIsoPref?.setOnPreferenceChangeListener { _, newValue ->
                isoPref?.isEnabled = newValue as? Boolean ?: true
                true
            }

            isoPref?.setOnPreferenceChangeListener { _, newValue ->
                val entered = (newValue as? String)?.trim().orEmpty()
                val clamped = CaptureSettings.normalizeIso(entered)
                if (clamped == null) {
                    Toast.makeText(context, "Please enter a valid ISO number.", Toast.LENGTH_SHORT).show()
                    false
                } else {
                    val normalizedValue = clamped.toString()
                    if (normalizedValue != entered) {
                        // Persist the corrected value ourselves and reject the original input.
                        isoPref.text = normalizedValue
                    }
                    Toast.makeText(context, "ISO set to $clamped", Toast.LENGTH_SHORT).show()
                    normalizedValue == entered
                }
            }
        }
    }
}
