package com.example.histogramapp

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.imgproc.Imgproc

object HistogramUtil {

    init {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            try {
                System.loadLibrary("opencv_java3")
            } catch (e2: UnsatisfiedLinkError) {
                throw RuntimeException(
                    "OpenCV 原生库加载失败。请确保:\n" +
                    "1. build.gradle 中已添加 OpenCV 依赖\n" +
                    "2. APK 中包含了原生 .so 文件 (arm64-v8a/armeabi-v7a)\n" +
                    "3. 如使用本地 .aar，将其置于 app/libs/ 目录",
                    e2
                )
            }
        }
    }

    private var srcMat: Mat? = null
    private var grayMat: Mat? = null
    private var histMat: Mat? = null

    private val channels: MatOfInt by lazy { MatOfInt(0) }
    private val histSize: MatOfInt by lazy { MatOfInt(256) }
    private val ranges: MatOfFloat by lazy { MatOfFloat(0f, 256f) }
    private val noMask: Mat by lazy { Mat() }

    private var histBitmap: Bitmap? = null
    private var canvas: Canvas? = null
    private var paint: Paint? = null

    private val histData = FloatArray(256)
    private val lock = Any()

    fun computeHistogram(bitmap: Bitmap): Pair<Bitmap, Long> = synchronized(lock) {
        val src = srcMat ?: Mat().also { srcMat = it }
        Utils.bitmapToMat(bitmap, src)

        val t0 = System.currentTimeMillis()

        val gray = grayMat ?: Mat().also { grayMat = it }
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val hist = histMat ?: Mat().also { histMat = it }
        Imgproc.calcHist(
            listOf(gray),
            channels,
            noMask,
            hist,
            histSize,
            ranges
        )

        Core.normalize(hist, hist, 0.0, 100.0, Core.NORM_MINMAX)

        val result = drawHistogram()

        val t1 = System.currentTimeMillis()

        return Pair(result, t1 - t0)
    }

    private fun drawHistogram(): Bitmap {
        val w = 256
        val h = 100

        var bmp = histBitmap
        var cvs = canvas
        var pnt = paint

        if (bmp == null || bmp.width != w || bmp.height != h) {
            bmp?.recycle()
            bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            cvs = Canvas(bmp)
            pnt = Paint().apply {
                color = Color.BLACK
                style = Paint.Style.FILL
                isAntiAlias = false
            }
            histBitmap = bmp
            canvas = cvs
            paint = pnt
        }

        val canvas = checkNotNull(cvs)
        val paint = checkNotNull(pnt)
        val bitmap = checkNotNull(bmp)

        canvas.drawColor(Color.WHITE)

        histMat?.get(0, 0, histData)

        for (x in 0 until w) {
            val bh = Math.round(histData[x]).coerceIn(0, h)
            if (bh > 0) {
                canvas.drawRect(
                    x.toFloat(),
                    (h - bh).toFloat(),
                    (x + 1).toFloat(),
                    h.toFloat(),
                    paint
                )
            }
        }

        return bitmap
    }
}