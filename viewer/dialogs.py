"""Help menu, About dialog, and How-to-use dialog."""

from __future__ import annotations

import os

from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QPixmap
from PyQt5.QtWidgets import (
    QAction, QDialog, QDialogButtonBox,
    QLabel, QLayout, QTextBrowser, QVBoxLayout,
)

_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "res")


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Kestrel")

        lay = QVBoxLayout(self)
        lay.setSizeConstraint(QLayout.SetFixedSize)  # non-resizable

        logo_path = os.path.join(_RES, "kestrel_logo.png")
        if os.path.exists(logo_path):
            lbl = QLabel()
            pix = QPixmap(logo_path).scaledToWidth(260, Qt.SmoothTransformation)
            lbl.setPixmap(pix)
            lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(lbl)

        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setFixedWidth(400)
        tb.setHtml("""
<h3 style="text-align:center; margin-top:4px;">Kestrel — 2DGS Native Viewer</h3>
<p>
A real-time, local viewer for <b>2D Gaussian Splatting</b> (surfel) models,
purpose-built for robotics and simulation sensor pipelines.
</p>
<p>
Renders RGB and depth in a single PyQt5 window. Designed to feed real-time 
frames straight into a sensor simulation stack.
</p>
<p>
Built on the 2D Gaussian Splatting CUDA rasterizer<br>
<i>Huang et al., 2024 — 2D Gaussian Splatting for Geometrically Accurate
Radiance Fields</i>.
</p>
<p><b>Author:</b> mlisi1 - Michele Lisi (michele.lisi@phd.unipi.it)</p>
""")
        tb.document().setTextWidth(390)
        tb.setFixedHeight(int(tb.document().size().height()) + 8)
        lay.addWidget(tb)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.accept)
        lay.addWidget(bb)


class HowToUseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How to use Kestrel")
        self.setMinimumSize(560, 540)

        tb = QTextBrowser()
        tb.setHtml("""
<h3>Loading a Model</h3>
<p>
<code>python main.py &lt;scene.ply&gt;</code> — load a single PLY file.<br>
<code>python main.py &lt;model_dir/&gt; [--iterations 30000]</code> — load from a
training output directory.<br>
Add <code>--build-index</code> to build the frustum-culling octree at startup.
</p>

<h3>Camera Controls</h3>
<table cellspacing="4">
<tr><td><b>LMB drag</b></td><td>Orbit around the look-at point</td></tr>
<tr><td><b>RMB drag</b></td><td>Pan the look-at point</td></tr>
<tr><td><b>Scroll wheel</b></td><td>Zoom (change distance)</td></tr>
<tr><td><b>W / S</b></td><td>Move look-at forward / backward</td></tr>
<tr><td><b>A / D</b></td><td>Strafe look-at left / right</td></tr>
<tr><td><b>E / Q</b></td><td>Translate camera + look-at up / down (world-up axis)</td></tr>
<tr><td><b>Arrow keys</b></td><td>FPS-style look in place (yaw and pitch)</td></tr>
<tr><td><b>R</b></td><td>Reset camera to origin</td></tr>
</table>
<p>Mouse inversion, speeds, and world-up axis are adjustable in the <i>Camera</i> panel.</p>

<h3>Frustum Culling</h3>
<p>
An octree index lets Kestrel skip off-screen splats before rasterisation, cutting
frame time significantly on large scenes. Click <b>Build Index</b> once; the index
is saved to <code>.kestrel/</code> and reloaded automatically on subsequent runs.
</p>

<h3>Compression</h3>
<p>
Four levels trade file size against quality:<br>
<b>L0</b> original &nbsp;·&nbsp; <b>L1</b> fp16 &nbsp;·&nbsp;
<b>L2</b> fp16 + SH degree 1 &nbsp;·&nbsp; <b>L3</b> fp16 + int8.<br>
Compressed PLYs are cached in <code>.kestrel/</code> and reused on the next load.
</p>

<h3>Render Types</h3>
<p>
The <i>Type</i> selector switches the output: <b>RGB</b>, <b>Edge</b>, <b>Alpha</b>,
<b>Normal</b>, <b>View-Normal</b>, <b>Depth</b>, <b>Depth-Distort</b>,
<b>Depth-to-Normal</b>, <b>Depth-to-Curvature</b>.
</p>

<h3>Split View</h3>
<p>
Enable <b>Split View</b> to compare two render types side by side.
<i>Left</i> and <i>Right</i> selectors choose what each half shows;
the <i>Split Pos</i> slider moves the divider.
</p>

<h3>Gaussian Model Parameters</h3>
<p>
<b>SH Degree</b> — reduce for a speed boost (higher spherical-harmonic bands skipped).<br>
<b>Opacity Threshold</b> — cull splats below this opacity value.<br>
<b>Sparsity</b> — render every Nth splat; 1 = all splats, 2 = half, etc.<br>
<b>Scale</b> — global splat size multiplier.<br>
<b>Pointcloud / disk mode</b> — render splat centres as points or oriented disks.
</p>

<h3>Crop Box</h3>
<p>
Enable and adjust the X / Y / Z min–max sliders to restrict rendering to an
axis-aligned region. Useful for isolating a specific object or floor plane.
</p>

<h3>Resolution &amp; Aspect Ratio</h3>
<p>
Choose a preset (360p – 4K) or type a custom width × height.
<b>Lock AR</b> keeps the aspect ratio fixed when changing either dimension.
The <i>Aspect</i> combo snaps to standard ratios (16:9, 4:3, …).
</p>
""")

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.accept)

        lay = QVBoxLayout(self)
        lay.addWidget(tb)
        lay.addWidget(bb)


class HelpMenu:
    """Attaches a Help menu to a QMainWindow's menu bar."""

    def __init__(self, window):
        menu      = window.menuBar().addMenu("Help")
        act_how   = QAction("How to use", window)
        act_about = QAction("About",      window)
        act_how.triggered.connect(  lambda: HowToUseDialog(window).exec_())
        act_about.triggered.connect(lambda: AboutDialog(window).exec_())
        menu.addAction(act_how)
        menu.addSeparator()
        menu.addAction(act_about)
