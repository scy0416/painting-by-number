# pip install PySide6
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QSize

renderer = QSvgRenderer("KakaoTalk_20251028_014558453_02.svg")
# 원하는 래스터 크기
size = QSize(1024, 1024)
img = QImage(size, QImage.Format_ARGB32)
img.fill(0)  # 투명 배경
p = QPainter(img)
renderer.render(p)
p.end()
img.save("out.png")
