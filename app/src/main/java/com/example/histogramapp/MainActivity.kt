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

    private fun computeAndDisplayHistogram(bitmap: Bitmap) {
        Thread {
            // 计算直方图，记录耗时
            val startTime = System.currentTimeMillis()

            // 获取像素数组（一次性拷贝，减少JNI调用开销）
            val width = bitmap.width
            val height = bitmap.height
            val pixelCount = width * height
            val pixels = IntArray(pixelCount)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            val histogram = HistogramUtil.computeHistogramParallel(pixels)

            val endTime = System.currentTimeMillis()
            val costMs = endTime - startTime

            // 生成直方图Bitmap
            val histogramBitmap = HistogramUtil.generateHistogramBitmap(histogram)

            runOnUiThread {
                histogramView.setImageBitmap(histogramBitmap)
                tvTime.text = "直方图生成耗时: ${costMs}ms"
            }
        }.start()
    }
}