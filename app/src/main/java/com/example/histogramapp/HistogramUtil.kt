package com.example.histogramapp

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.util.concurrent.atomic.AtomicIntegerArray

object HistogramUtil {

    // ========== 核心：多线程并行计算直方图 ==========

    /**
     * 使用多线程并行计算直方图
     * @param pixels 图像像素数组（ARGB格式）
     * @return 长度为256的IntArray，每个元素对应0-255灰度值的像素计数
     */
    fun computeHistogramParallel(pixels: IntArray): IntArray {
        val histogram = AtomicIntegerArray(256)
        val numThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        val chunkSize = pixels.size / numThreads

        val threads = Array(numThreads) { threadIndex ->
            Thread {
                val start = threadIndex * chunkSize
                val end = if (threadIndex == numThreads - 1) pixels.size else start + chunkSize

                // 局部计数数组，减少AtomicIntegerArray的竞争
                val localHist = IntArray(256)

                for (i in start until end) {
                    val pixel = pixels[i]

                    // 提取RGB分量
                    val r = (pixel shr 16) and 0xFF
                    val g = (pixel shr 8) and 0xFF
                    val b = pixel and 0xFF

                    // 灰度化公式：gray = red*0.299 + green*0.587 + blue*0.114
                    val gray = (r * 299 + g * 587 + b * 114) / 1000
                    val clampedGray = gray.coerceIn(0, 255)

                    localHist[clampedGray]++
                }

                // 合并到全局AtomicIntegerArray
                for (i in 0..255) {
                    if (localHist[i] > 0) {
                        histogram.addAndGet(i, localHist[i])
                    }
                }
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }

        // 转回普通IntArray返回
        val result = IntArray(256)
        for (i in 0..255) {
            result[i] = histogram[i]
        }
        return result
    }

    // ========== 生成256x100的黑白直方图Bitmap ==========

    /**
     * 将直方图数据归一化并绘制为256x100的黑白位图
     * @param histogram 长度为256的直方图数据
     * @return 256x100的Bitmap，白色背景，黑色柱状
     */
    fun generateHistogramBitmap(histogram: IntArray): Bitmap {
        val width = 256
        val height = 100

        // 找到最大值用于归一化
        val maxCount = histogram.maxOrNull() ?: 1

        // 创建Bitmap
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        // 白色背景
        canvas.drawColor(Color.WHITE)

        val paint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
        }

        // 逐列绘制
        for (x in 0 until width) {
            val count = histogram[x]
            // 归一化到0-100
            val barHeight = if (maxCount > 0) {
                ((count.toFloat() / maxCount) * height).toInt()
            } else {
                0
            }

            // 从底部向上画
            val top = height - barHeight
            canvas.drawRect(
                    x.toFloat(),
                    top.toFloat(),
                    (x + 1).toFloat(),
                    height.toFloat(),
                    paint
            )
        }

        return bitmap
    }
}