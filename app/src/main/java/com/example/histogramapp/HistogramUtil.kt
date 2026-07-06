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

/**
 * 基于 OpenCV 原生 C++ 实现的图像直方图工具类。
 *
 * 流水线: Bitmap → Mat(RGBA) → cvtColor(GRAY) → calcHist → normalize → 绘制
 *
 * 所有 OpenCV Mat、Android Bitmap/Canvas/Paint 均复用，避免 GC 压力。
 *
 * 灰度公式 (OpenCV COLOR_RGBA2GRAY 内部使用):
 *   gray = 0.299*R + 0.587*G + 0.114*B
 * 与题目要求完全一致 (ITU-R BT.601)。
 */
object HistogramUtil {

    // ==================== OpenCV 初始化 ====================

    init {
        // 加载 OpenCV 原生库。库名因版本而异:
        //   OpenCV 4.x → opencv_java4
        //   OpenCV 3.x → opencv_java3
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

    // ==================== 可复用 Mat 对象 ====================

    /** 输入 RGBA 图像 (4 通道)，尺寸随输入图片变化 */
    private var srcMat: Mat? = null

    /** 灰度图像 (单通道 CV_8UC1)，尺寸随输入图片变化 */
    private var grayMat: Mat? = null

    /** 直方图结果 (256×1 CV_32F)，尺寸固定 — 始终复用 */
    private var histMat: Mat? = null

    // ==================== 可复用 calcHist 参数 (尺寸固定，永不变化) ====================

    /** 通道索引: 0 = 灰度图的唯一通道 */
    private val channels: MatOfInt by lazy { MatOfInt(0) }

    /** 直方图 bin 数量: 256 */
    private val histSize: MatOfInt by lazy { MatOfInt(256) }

    /** 像素值范围: [0, 256) */
    private val ranges: MatOfFloat by lazy { MatOfFloat(0f, 256f) }

    /** 空掩码: 统计全部像素 */
    private val noMask: Mat by lazy { Mat() }

    // ==================== 可复用绘制对象 ====================

    /** 输出的 256×100 直方图 Bitmap */
    private var histBitmap: Bitmap? = null

    /** Canvas 绑定到 histBitmap */
    private var canvas: Canvas? = null

    /** 黑色填充画笔，关闭抗锯齿 */
    private var paint: Paint? = null

    /** 直方图数据读取缓冲区 (256 个 float)，单次 JNI 调用读取全部 */
    private val histData = FloatArray(256)

    /** 对象锁，防止并发访问复用对象 */
    private val lock = Any()

    // ==================== 公开 API ====================

    /**
     * 使用 OpenCV 原生实现计算 256 级灰度直方图并生成 256×100 黑白可视化。
     *
     * 计时区间仅覆盖: 灰度转换 → calcHist → normalize → 绘制直方图 Bitmap
     *
     * @param bitmap 输入彩色图片 (任意尺寸，不缩放、不采样)
     * @return Pair(256×100 直方图 Bitmap, 耗时 ms)
     */
    fun computeHistogram(bitmap: Bitmap): Pair<Bitmap, Long> = synchronized(lock) {
        // ---- 1. Bitmap → RGBA Mat (不计入耗时，属于输入准备) ----
        if (srcMat == null) srcMat = Mat()
        Utils.bitmapToMat(bitmap, srcMat!!)

        // ========== 开始计时 ==========
        val t0 = System.currentTimeMillis()

        // ---- 2. RGBA → Grayscale (OpenCV 原生 cvtColor) ----
        if (grayMat == null) grayMat = Mat()
        Imgproc.cvtColor(srcMat!!, grayMat!!, Imgproc.COLOR_RGBA2GRAY)

        // ---- 3. 直方图统计 (OpenCV 原生 calcHist，遍历全图像素) ----
        if (histMat == null) histMat = Mat()
        Imgproc.calcHist(
            listOf(grayMat!!),
            channels,
            noMask,
            histMat!!,
            histSize,
            ranges
        )

        // ---- 4. 归一化到 [0, 100] (OpenCV 原生 normalize) ----
        Core.normalize(histMat!!, histMat!!, 0.0, 100.0, Core.NORM_MINMAX)

        // ---- 5. 绘制 256×100 黑白直方图 Bitmap ----
        val result = drawHistogram()

        // ========== 结束计时 ==========
        val t1 = System.currentTimeMillis()

        return Pair(result, t1 - t0)
    }

    // ==================== 直方图绘制 ====================

    /**
     * 将归一化后的直方图数据绘制为 256×100 黑白 Bitmap。
     *
     * 复用策略:
     *  - Bitmap: 尺寸固定 (256×100)，一次创建终生复用
     *  - Canvas: 绑定到复用 Bitmap
     *  - Paint: 对象池式复用
     *  - histData: 预分配缓冲区，避免每次读取时 FloatArray 分配
     */
    private fun drawHistogram(): Bitmap {
        val w = 256
        val h = 100

        // 按需创建或复用 Bitmap / Canvas / Paint
        if (histBitmap == null || histBitmap!!.width != w || histBitmap!!.height != h) {
            histBitmap?.recycle()
            histBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            canvas = Canvas(histBitmap!!)
            paint = Paint().apply {
                color = Color.BLACK
                style = Paint.Style.FILL
                isAntiAlias = false   // 1px 宽柱体无需抗锯齿
            }
        }

        // 白色背景 (一次性填充)
        canvas!!.drawColor(Color.WHITE)

        // 单次 JNI 调用读取全部 256 个归一化直方图值
        histMat!!.get(0, 0, histData)

        // 逐列从底部向上绘制黑色柱体
        for (x in 0 until w) {
            val bh = histData[x].toInt().coerceIn(0, h)
            if (bh > 0) {
                canvas!!.drawRect(
                    x.toFloat(),
                    (h - bh).toFloat(),
                    (x + 1).toFloat(),
                    h.toFloat(),
                    paint!!
                )
            }
        }

        return histBitmap!!
    }
}
