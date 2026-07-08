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

/**
 * HistogramApp 主界面 Activity。
 *
 * ## 界面布局
 * ```
 * ┌──────────────────────────┐
 * │  [选择图片]               │  ← btnSelect (Button)
 * ├──────────────────────────┤
 * │                          │
 * │  原图预览                 │  ← imageView (ImageView, fitCenter)
 * │                          │
 * ├──────────────────────────┤
 * │  直方图生成耗时: 23ms     │  ← tvTime (TextView, 居中加粗)
 * ├──────────────────────────┤
 * │  直方图 (256×100)        │  ← 标签 (TextView)
 * │  ▊▊ ▊▊▊ ▊▊ ▊▊▊▊        │  ← histogramView (ImageView, fitXY)
 * └──────────────────────────┘
 * ```
 *
 * ## 线程模型
 * - **UI 线程 (Main)**: 图片选择、界面渲染、Toast 提示
 * - **后台线程 (Thread-*)**: 调用 [HistogramUtil.computeHistogram] 执行 OpenCV 管线
 * - **OpenCV 内部线程池 (TBB)**: cvtColor / calcHist 的 NEON 并行执行
 *
 * ## 错误处理
 * - 图片读取失败 → Toast 提示 + 界面保持不变
 * - OpenCV 计算失败 → Toast 提示 + 直方图区域清空 + 耗时重置为 "--"
 *
 * ## 生命周期
 * - [onCreate] 执行标准初始化: setContentView + findViewById + 事件绑定
 * - 无 onPause/onResume 特殊处理 (直方图计算是瞬时操作)
 * - 无 onDestroy 资源释放 ([HistogramUtil] 为单例，生命周期跟随进程)
 *
 * @author HistogramApp Team
 * @since 1.0
 */
class MainActivity : AppCompatActivity() {

    // =========================================================================
    //  UI 组件
    // =========================================================================

    /** 原图预览 — 显示用户从相册选择的图片，scaleType = fitCenter */
    private lateinit var imageView: ImageView

    /** 直方图显示 — 显示 HistogramUtil 生成的 256×100 黑白直方图，scaleType = fitXY */
    private lateinit var histogramView: ImageView

    /** 耗时显示 — 格式: "直方图生成耗时: XXms"，加粗居中 */
    private lateinit var tvTime: TextView

    /** 选择图片按钮 — 点击触发系统相册选择器 */
    private lateinit var btnSelect: Button

    // =========================================================================
    //  图片选择器
    // =========================================================================

    /**
     * 系统图片选择器 (通过 Activity Result API 注册)。
     *
     * ## 替代方案说明
     * 使用 `registerForActivityResult(ActivityResultContracts.GetContent())`
     * 替代已废弃的 `startActivityForResult + onActivityResult` 模式:
     * - 无需手动管理 requestCode
     * - 回调闭包内可直接访问 UI 组件
     * - Google 推荐的现代 Android 开发方式
     *
     * ## 流程
     * 1. 用户点击「选择图片」→ `pickImage.launch("image/*")`
     * 2. 系统相册选择器弹出
     * 3. 用户选择图片 → ContentProvider 返回 Uri
     * 4. 通过 ContentResolver 打开 InputStream 读取图片数据
     * 5. BitmapFactory.decodeStream 解码为 Bitmap
     * 6. 显示原图 + 触发直方图计算
     */
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()  // 选择任意类型内容，此处限定 "image/*"
    ) { uri: Uri? ->
        // uri 为 null 表示用户取消了选择操作
        uri?.let {
            try {
                // ContentResolver.openInputStream:
                //   通过 ContentProvider (通常是 MediaStore) 读取图片文件的输入流
                val inputStream: InputStream? = contentResolver.openInputStream(it)

                // BitmapFactory.decodeStream:
                //   从 InputStream 解码图片为 Bitmap 对象
                //   输出格式: ARGB_8888 (4 字节/像素, R+G+B+A 各 8 位)
                //   如果原图色彩空间不同 (如 RGB_565)，decodeStream 会自动转换
                val bitmap = BitmapFactory.decodeStream(inputStream)

                // InputStream 使用完毕后立即关闭，释放文件描述符
                inputStream?.close()

                if (bitmap != null) {
                    // 主线程操作: 显示原图预览
                    imageView.setImageBitmap(bitmap)

                    // 异步执行: 在后台线程计算直方图
                    computeAndDisplayHistogram(bitmap)
                } else {
                    // BitmapFactory 解码失败 (如文件损坏、格式不支持)
                    Toast.makeText(this, "图片解码失败", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                // 捕获所有异常: 文件读取失败、权限不足、OOM 等
                Toast.makeText(this, "读取图片失败: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // =========================================================================
    //  生命周期
    // =========================================================================

    /**
     * Activity 创建入口 — 执行一次性初始化。
     *
     * ## 初始化顺序
     * 1. `super.onCreate()` — 必须首先调用
     * 2. `setContentView()` — 加载 XML 布局，构建 View 树
     * 3. `findViewById()` — 获取 UI 组件引用 (lateinit var 延迟初始化)
     * 4. `setOnClickListener()` — 绑定图片选择按钮点击事件
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        // 调用父类 onCreate — Android 标准初始化流程
        super.onCreate(savedInstanceState)

        // 加载布局文件: res/layout/activity_main.xml
        // LinearLayout (vertical) 包含:
        //   - Button (选择图片)
        //   - ImageView (原图预览)
        //   - TextView (耗时显示)
        //   - ImageView (直方图显示)
        setContentView(R.layout.activity_main)

        // findViewById — 通过 XML 中声明的 android:id 获取 View 引用
        // 注意: 如果 XML 中缺少对应 id，此处将抛出 NPE (符合 fail-fast 原则)
        imageView = findViewById(R.id.imageView)
        histogramView = findViewById(R.id.histogramView)
        tvTime = findViewById(R.id.tvTime)
        btnSelect = findViewById(R.id.btnSelect)

        // 点击「选择图片」→ 启动系统图片选择器
        // launch("image/*"): MIME 类型过滤，仅显示图片文件 (JPEG, PNG, GIF, WebP 等)
        btnSelect.setOnClickListener {
            pickImage.launch("image/*")
        }
    }

    // =========================================================================
    //  直方图计算与显示
    // =========================================================================

    /**
     * 后台线程执行 OpenCV 直方图计算，UI 线程更新界面。
     *
     * ## 执行流程
     * ```
     * UI 线程 (当前)
     *   │
     *   ├─ Thread { ... }.start()
     *   │     │
     *   │     ├─ HistogramUtil.computeHistogram(bitmap)
     *   │     │     ├─ Utils.bitmapToMat()         (不计时)
     *   │     │     ├─ Imgproc.cvtColor()          ←── 计时开始
     *   │     │     ├─ Imgproc.calcHist()
     *   │     │     ├─ Core.normalize()
     *   │     │     └─ Canvas 绘制 256×100 Bitmap  ←── 计时结束
     *   │     │
     *   │     └─ runOnUiThread { ... }
     *   │           ├─ histogramView.setImageBitmap()
     *   │           └─ tvTime.text = "...ms"
     *   │
     *   └─ return (UI 线程不阻塞，立即返回)
     * ```
     *
     * ## 计时规则
     * 仅统计 `HistogramUtil.computeHistogram()` 内部的四个核心步骤:
     * 灰度转换 → 直方图统计 → 归一化 → 位图绘制。
     * Bitmap→Mat 转换不计入 (属输入准备)。
     *
     * ## 异常处理
     * 如果 OpenCV 原生库加载失败或计算过程异常:
     * - 直方图区域清空 (setImageBitmap(null))
     * - 耗时显示重置为 "-- ms"
     * - Toast 提示用户重试
     *
     * @param bitmap 用户选择的原始图片 (ARGB_8888 格式)
     */
    private fun computeAndDisplayHistogram(bitmap: Bitmap) {
        // 创建并启动后台线程。
        // 使用匿名 Thread 而非线程池:
        //   - 直方图计算是一次性操作 (非频繁重复任务)
        //   - OpenCV 内部已使用 TBB 线程池做并行化
        //   - 匿名线程在 run() 结束后自动销毁，无资源泄漏
        Thread {
            try {
                // HistogramUtil.computeHistogram 是同步调用:
                //   传入 Bitmap，阻塞等待 OpenCV 管线完成，返回结果。
                //
                // 解构声明 (destructuring declaration):
                //   Pair<Bitmap, Long> → (histogramBitmap, costMs)
                //
                // histogramBitmap: 256×100 ARGB_8888，白色背景黑色柱体
                // costMs: 核心管线耗时 (长整型，单位毫秒)
                val (histogramBitmap, costMs) = HistogramUtil.computeHistogram(bitmap)

                // runOnUiThread: 切换回 UI 线程更新界面
                // View 操作必须在 UI 线程执行，否则抛出 CalledFromWrongThreadException
                runOnUiThread {
                    // 设置直方图 ImageView 的显示内容
                    // 由于 256×100 是固定小尺寸，setImageBitmap 瞬间完成
                    histogramView.setImageBitmap(histogramBitmap)

                    // 更新耗时文字
                    // 格式示例: "直方图生成耗时: 23ms"
                    tvTime.text = "直方图生成耗时: ${costMs}ms"
                }
            } catch (e: Exception) {
                // 异常场景:
                //   1. UnsatisfiedLinkError → OpenCV .so 未正确打包或未加载
                //   2. OutOfMemoryError     → 图片过大 (通常 > 100MP)
                //   3. IllegalArgumentException → Bitmap 格式不被 OpenCV 支持
                //
                // 处理方式: UI 线程显示错误提示并重置界面状态
                runOnUiThread {
                    Toast.makeText(this, "图像处理失败，请重试", Toast.LENGTH_LONG).show()
                    histogramView.setImageBitmap(null)       // 清空直方图区域
                    tvTime.text = "直方图生成耗时: -- ms"     // 重置耗时显示
                }
            }
        }.start()  // start() 后立即返回，不阻塞 UI 线程
    }
}
