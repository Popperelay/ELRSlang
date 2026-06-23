package dev.elrslang;

import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.TextView;

public final class SmokeActivity extends Activity {
    static {
        System.loadLibrary("elrslang_smoke");
    }

    private static native String nativeSmokeStatus(AssetManager assetManager);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        TextView view = new TextView(this);
        view.setText(nativeSmokeStatus(getAssets()));
        view.setTextSize(16.0f);
        int padding = 32;
        view.setPadding(padding, padding, padding, padding);
        setContentView(view);
    }
}
