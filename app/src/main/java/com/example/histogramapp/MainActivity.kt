package com.example.histogramapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var histogramView: ImageView
    private lateinit var tvTime: TextView
    private lateinit var btnSelect: Button

    // 选择图片
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                val inputStream: InputStream? = contentResolver.openInputStream(it)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap != null) {
                    imageView.setImageBitmap(bitmap)
                    computeAndDisplayHistogram(bitmap)
                }
            } catch (e: Exception) {
                Toast.makeText(this, "读取图片失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        histogramView = findViewById(R.id.histogramView)
        tvTime = findViewById(R.id.tvTime)
        btnSelect = findViewById(R.id.btnSelect)

        btnSelect.setOnClickListener {
            pickImage.launch("image/*")
        }
    }

    /**
     * 后台线程执行 OpenCV 直方图计算，UI 线程仅负责界面更新。
     *
     * 计时区间: 灰度转换 → calcHist → normalize → 绘制直方图
     * (Bitmap → Mat 转换不计入耗时)
     */
    private fun computeAndDisplayHistogram(bitmap: Bitmap) {
        Thread {
            // HistogramUtil.computeHistogram 内部完成全部管线:
            //   Bitmap→Mat → cvtColor → calcHist → normalize → 绘制 256×100 Bitmap
            // 返回 (直方图 Bitmap, 耗时 ms)
            val (histogramBitmap, costMs) = HistogramUtil.computeHistogram(bitmap)

            runOnUiThread {
                histogramView.setImageBitmap(histogramBitmap)
                tvTime.text = "直方图生成耗时: ${costMs}ms"
            }
        }.start()
    }
}
