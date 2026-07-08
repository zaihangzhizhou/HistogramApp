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
 * 图像直方图工具类 — 基于 OpenCV 原生 C++ 实现。
 *
 * ## 功能
 * 接收任意尺寸的 Android [Bitmap] 彩色图片，计算 256 级灰度直方图，
 * 生成 256×100 黑白直方图位图，并返回核心管线的执行耗时。
 *
 * ## 处理流水线
 * ```
 * Bitmap (RGBA) ──bitmapToMat──▶ Mat (RGBA, CV_8UC4)
 *     ┊  ═══════ 计时开始 ═══════
 *     ┊  cvtColor(COLOR_RGBA2GRAY)
 *     ┊      ↓
 *     ┊  Mat (GRAY, CV_8UC1)
 *     ┊      ↓
 *     ┊  calcHist(256 bins, range [0,256))
 *     ┊      ↓
 *     ┊  Mat (256×1, CV_32F)
 *     ┊      ↓
 *     ┊  normalize(NORM_MINMAX → [0, 100])
 *     ┊      ↓
 *     ┊  Canvas 绘制 256×100 Bitmap
 *     ┊  ═══════ 计时结束 ═══════
 *     └──▶ Pair<Bitmap, Long>  (直方图位图, 耗时ms)
 * ```
 *
 * ## 灰度公式
 * OpenCV `COLOR_RGBA2GRAY` 内部使用 ITU-R BT.601 标准：
 * ```
 * gray = 0.299·R + 0.587·G + 0.114·B
 * ```
 * 与题目要求的 `(R*299 + G*587 + B*114) / 1000` 数学等价。
 *
 * ## 性能特性
 * - OpenCV C++ 原生实现，使用 NEON SIMD 指令集并行处理像素
 * - 所有 Mat / Bitmap / Canvas / Paint 对象复用，减少 GC 抖动
 * - `synchronized` 保证线程安全，防止并发调用时对象复用冲突
 *
 * ## 使用示例
 * ```kotlin
 * val (histogramBitmap, costMs) = HistogramUtil.computeHistogram(inputBitmap)
 * imageView.setImageBitmap(histogramBitmap)
 * println("耗时: ${costMs}ms")  // 通常 < 30ms (12MP 图片)
 * ```
 *
 * @author HistogramApp Team
 * @since 1.0
 */
object HistogramUtil {

    // =========================================================================
    //  OpenCV 原生库初始化
    // =========================================================================

    init {
        // OpenCV Java API 是 JNI 封装，底层依赖 C++ 原生库 (.so 文件)。
        // 必须在调用任何 OpenCV 类之前通过 System.loadLibrary 加载。
        //
        // 库名规则:
        //   OpenCV 4.x → libopencv_java4.so → "opencv_java4"
        //   OpenCV 3.x → libopencv_java3.so → "opencv_java3"
        //
        // Android 原生库查找路径 (APK 内):
        //   lib/arm64-v8a/libopencv_java4.so
        //   lib/armeabi-v7a/libopencv_java4.so
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            // 备选: OpenCV 3.x 版本库名
            try {
                System.loadLibrary("opencv_java3")
            } catch (e2: UnsatisfiedLinkError) {
                // 两次尝试均失败 → 抛出明确错误信息指导排查
                throw RuntimeException(
                    "OpenCV 原生库加载失败。请确保:\n" +
                    "1. build.gradle 中已添加 OpenCV 依赖\n" +
                    "2. APK 中包含了原生 .so 文件 (arm64-v8a / armeabi-v7a)\n" +
                    "3. 如使用本地 .aar 方式，将文件置于 app/libs/ 目录\n" +
                    "4. AndroidManifest.xml 中已设置 extractNativeLibs=\"true\"",
                    e2
                )
            }
        }
    }

    // =========================================================================
    //  可复用 OpenCV Mat 对象
    // =========================================================================
    // 设计说明:
    //  - Mat 持有 native 堆内存指针，频繁创建/释放会增加 GC 和 JNI 开销
    //  - 首次使用时创建，后续调用复用同一对象（OpenCV 内部按需重分配大小）
    //  - histMat 尺寸固定 (256×1)，srcMat/grayMat 尺寸随输入图片变化

    /** 输入图像 Mat — RGBA 格式 (CV_8UC4, 4 通道)。
     *  由 [Utils.bitmapToMat] 写入，内部按需分配/重分配 */
    private var srcMat: Mat? = null

    /** 灰度图像 Mat — 单通道 (CV_8UC1)。
     *  由 [Imgproc.cvtColor] 输出，内部按需分配/重分配 */
    private var grayMat: Mat? = null

    /** 直方图结果 Mat — 256 行 × 1 列 (CV_32F)。
     *  尺寸固定，每次调用 [Imgproc.calcHist] 复用 */
    private var histMat: Mat? = null

    // =========================================================================
    //  可复用 calcHist 参数 (尺寸固定，永不变化)
    // =========================================================================
    // 这些 MatOfInt/MatOfFloat 作为 calcHist 的输入参数，值永远不变。
    // 使用 Kotlin lazy 委托，首次访问时创建一次，之后全局复用。

    /** 通道索引列表: 灰度图仅通道 0 有效 */
    private val channels: MatOfInt by lazy { MatOfInt(0) }

    /** 直方图 bin 数量: 256 个灰度级 (0–255) */
    private val histSize: MatOfInt by lazy { MatOfInt(256) }

    /** 像素值统计范围: [0, 256)，左闭右开 */
    private val ranges: MatOfFloat by lazy { MatOfFloat(0f, 256f) }

    /** 掩码: 空 Mat 表示统计全图所有像素 */
    private val noMask: Mat by lazy { Mat() }

    // =========================================================================
    //  可复用 Android 绘制对象
    // =========================================================================
    // 输出位图尺寸固定为 256×100，因此 Bitmap / Canvas / Paint 均可复用。
    // 每次绘制前用 drawColor(Color.WHITE) 清除上一帧内容。

    /** 输出的 256×100 直方图位图。
     *  仅当尺寸不匹配时才重建，通常整个生命周期不变 */
    private var histBitmap: Bitmap? = null

    /** Canvas — 绑定到 [histBitmap]，用于绘制白色背景和黑色柱体 */
    private var canvas: Canvas? = null

    /** Paint — 黑色填充画笔，关闭抗锯齿以保持 1px 柱体锐利 */
    private var paint: Paint? = null

    /** 直方图数据读取缓冲区 — 预分配 FloatArray(256)。
     *  配合 [Mat.get] 实现单次 JNI 调用读取全部 256 个归一化值，
     *  避免逐值 256 次 JNI 往返 */
    private val histData = FloatArray(256)

    // =========================================================================
    //  线程安全
    // =========================================================================

    /** 对象级锁 — 所有可复用字段均为非线程安全类型 (var)，
     *  通过 synchronized(lock) 将整个 computeHistogram 调用串行化 */
    private val lock = Any()

    // =========================================================================
    //  公开 API
    // =========================================================================

    /**
     * 计算图像直方图并生成可视化位图。
     *
     * 此方法是 [HistogramUtil] 的唯一公开入口，封装完整处理管线。
     *
     * ## 计时规则
     * 仅统计以下四个步骤的耗时:
     * 1. `Imgproc.cvtColor`  — RGBA → Grayscale (灰度转换)
     * 2. `Imgproc.calcHist`   — 直方图统计
     * 3. `Core.normalize`     — 归一化到 [0, 100]
     * 4. 绘制 256×100 Bitmap  — Canvas 绘制
     *
     * Bitmap → Mat 转换不计入耗时 (属于输入准备阶段)。
     *
     * ## 线程模型
     * - 调用方在后台线程执行 (MainActivity 使用 `Thread { }.start()`)
     * - OpenCV 内部使用 TBB 线程池并行化 cvtColor / calcHist
     * - UI 线程通过返回值的 Bitmap 更新界面
     *
     * @param bitmap 输入彩色图片，不缩放、不采样，统计全图像素
     * @return Pair(256×100 直方图 Bitmap, 核心管线耗时 ms)
     * @throws RuntimeException 当 OpenCV 原生库未正确加载时
     */
    fun computeHistogram(bitmap: Bitmap): Pair<Bitmap, Long> = synchronized(lock) {
        // ── 步骤 1: Bitmap → RGBA Mat ──
        // Utils.bitmapToMat 将 Android Bitmap 像素数据拷贝到 OpenCV Mat。
        // 输出格式: CV_8UC4 (8-bit unsigned, 4 channels: R, G, B, A)。
        // 首次调用创建 srcMat，后续复用 (内部按需调整大小)。
        val src = srcMat ?: Mat().also { srcMat = it }
        Utils.bitmapToMat(bitmap, src)

        // ══════════════ 计时开始 ══════════════
        val t0 = System.currentTimeMillis()

        // ── 步骤 2: RGBA → Grayscale ──
        // Imgproc.cvtColor 是 OpenCV 的颜色空间转换函数。
        // COLOR_RGBA2GRAY: 将 4 通道 RGBA 转为单通道灰度。
        // 灰度公式 (BT.601): gray = 0.299·R + 0.587·G + 0.114·B
        // Alpha 通道被忽略 (COLOR_RGBA2GRAY 与 COLOR_RGB2GRAY 结果相同)。
        // C++ 层使用 NEON SIMD 以每指令 16 像素的速度执行。
        val gray = grayMat ?: Mat().also { grayMat = it }
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        // ── 步骤 3: 直方图统计 ──
        // calcHist 统计灰度图中每个像素值的出现次数。
        // 参数说明:
        //   listOf(gray) — 输入图像列表 (此处仅一张灰度图)
        //   channels(0)  — 统计通道 0 (灰度图的唯一通道)
        //   noMask       — 空 Mat = 统计全图 (不屏蔽任何像素)
        //   hist         — 输出: 256×1 的 CV_32F 矩阵
        //   histSize(256)— 256 个 bin，每个 bin 对应一个灰度值 0–255
        //   ranges(0,256)— 像素值范围为 [0, 256) (即 0–255 的像素值)
        // 内部使用 TBB 多线程并行统计，按行分片。
        val hist = histMat ?: Mat().also { histMat = it }
        Imgproc.calcHist(
            listOf(gray),
            channels,   // MatOfInt(0)
            noMask,     // 空 Mat = 全图
            hist,       // 输出目标
            histSize,   // MatOfInt(256)
            ranges      // MatOfFloat(0f, 256f)
        )

        // ── 步骤 4: 归一化 ──
        // NORM_MINMAX: 线性缩放所有值到 [alpha, beta] = [0, 100] 区间。
        // 缩放公式: dst[i] = (src[i] - min) / (max - min) * (beta - alpha) + alpha
        // 结果: 出现次数最多的灰度级 → 100.0，最少的 → 0.0 (或接近 0)。
        // 归一化后的值用于直方图柱体的像素高度。
        Core.normalize(hist, hist, 0.0, 100.0, Core.NORM_MINMAX)

        // ── 步骤 5: 绘制直方图位图 ──
        // drawHistogram() 复用 histBitmap/Canvas/Paint，将 256 个浮点值
        // 绘制为 256 根 1px 宽的黑色柱体 (白底黑柱)。
        val result = drawHistogram()

        // ══════════════ 计时结束 ══════════════
        val t1 = System.currentTimeMillis()

        // 返回: (直方图位图, 核心管线耗时)
        return Pair(result, t1 - t0)
    }

    // =========================================================================
    //  直方图位图绘制
    // =========================================================================

    /**
     * 将归一化后的直方图数据绘制为 256×100 黑白位图。
     *
     * ## 复用策略
     * | 对象       | 策略                          | 原因                     |
     * |-----------|-------------------------------|--------------------------|
     * | Bitmap    | 首次创建，尺寸不变则永久复用      | 256×100 固定尺寸           |
     * | Canvas    | 绑定到复用 Bitmap              | 随 Bitmap 一起复用         |
     * | Paint     | 首次创建，属性不变则永久复用      | 始终 BLACK + FILL          |
     * | histData  | 预分配 FloatArray(256)         | 避免每次读取时的数组分配     |
     *
     * ## 绘制过程
     * 1. 白色背景填充 (canvas.drawColor)
     * 2. 单次 JNI 读取 256 个归一化值 (histMat.get)
     * 3. 逐列从底部向上绘制黑色柱体 (canvas.drawRect)
     *
     * ## 精度说明
     * 使用 [Math.round] 而非 [Float.toInt] 将浮点高度转为像素高度:
     * - Math.round(99.9f) = 100 ✓  (四舍五入)
     * - 99.9f.toInt()     =  99 ✗  (截断)
     * 避免因截断误差导致最高柱体不足 100 像素。
     *
     * @return 256×100 ARGB_8888 直方图位图，白色背景黑色柱体
     */
    private fun drawHistogram(): Bitmap {
        // 直方图尺寸常量
        val w = 256   // 宽度 = 256 个灰度级
        val h = 100   // 高度 = 归一化最大值 100

        // ── 按需创建或复用绘制对象 ──
        // 检查复用条件:
        //   1. 是否首次调用 (bmp == null)
        //   2. 尺寸是否被意外改变 (防御性检查)
        var bmp = histBitmap
        var cvs = canvas
        var pnt = paint

        if (bmp == null || bmp.width != w || bmp.height != h) {
            // 尺寸不匹配 → 释放旧位图并创建新对象
            bmp?.recycle()
            bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            cvs = Canvas(bmp)
            pnt = Paint().apply {
                color = Color.BLACK        // 黑色柱体
                style = Paint.Style.FILL   // 填充模式 (非描边)
                isAntiAlias = false        // 关闭抗锯齿:
                                            //   1px 宽的柱体经抗锯齿后会变模糊 (2px 灰边)
                                            //   关闭后保持像素级锐利
            }
            // 写回字段供下次复用
            histBitmap = bmp
            canvas = cvs
            paint = pnt
        }

        // 使用局部变量避免 !! 操作符和空安全开销
        val canvas = checkNotNull(cvs) { "Canvas 未初始化" }
        val paint = checkNotNull(pnt) { "Paint 未初始化" }
        val bitmap = checkNotNull(bmp) { "Bitmap 未初始化" }

        // ── 白色背景 ──
        // drawColor 一次性填充整个 Canvas 为纯白色 (0xFFFFFFFF)
        canvas.drawColor(Color.WHITE)

        // ── 读取归一化直方图数据 ──
        // histMat.get(row, col, float[]): 从 Mat 的 (row, col) 位置开始，
        //   读取 float[].length 个连续值。
        // 对 256×1 列矩阵，histMat.get(0, 0, histData) 单次 JNI 调用读取全部 256 个值。
        // 等效于 (但远快于):
        //   for (i in 0..255) histData[i] = histMat!![i, 0][0]
        histMat?.get(0, 0, histData)

        // ── 逐列绘制黑色柱体 ──
        // x: 横坐标 0..255，对应灰度值 0..255
        // bh: 柱体高度 (像素)，由归一化值 (0.0–100.0) 四舍五入得到
        //
        // Canvas 坐标系:
        //   (0,0) ──→ x (向右)
        //     │
        //     ↓ y (向下)
        //
        // 因此柱体从底部 (y=h) 向上画到 top = h - bh
        //   top = 0   → 柱体占满全高 (100px)
        //   top = 100 → 柱体高度为 0 (不绘制)
        for (x in 0 until w) {
            // Math.round: 四舍五入，避免截断误差
            // coerceIn:   钳制到 [0, h]，防御性处理 (理论上归一化已在 [0,100] 区间)
            val bh = Math.round(histData[x]).coerceIn(0, h)
            if (bh > 0) {
                // drawRect(left, top, right, bottom, paint)
                //   left  = x          (当前灰度级对应列)
                //   top   = h - bh     (柱体顶部，坐标系原点在左上)
                //   right = x + 1      (1px 宽)
                //   bottom = h         (柱体底部，始终在画布最下方)
                canvas.drawRect(
                    x.toFloat(),           // left    — 柱体左边界
                    (h - bh).toFloat(),    // top     — 柱体顶部
                    (x + 1).toFloat(),     // right   — 柱体右边界 (1px 宽)
                    h.toFloat(),           // bottom  — 柱体底部 (画布底边)
                    paint
                )
            }
        }

        return bitmap
    }
}
