# 4: Dateidialog

import sys

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5 import uic




# ---------------------------------------------

class Actions(QDialog):
    """
    Simple dialog that consists of a Progress Bar and a Button.
    Clicking on the button results in the start of a timer and
    updates the progress bar.
    In Version 3c kommt noch die variable Schrittweite hinzu.

    Wenn Sie auf die Schaltfläche self.onButtonClick klicken,
    wird self.onButtonClick ausgeführt und der Thread gestartet. 
    Der Thread wird mit .start() gestartet. Es sollte auch beachtet werden, 
    dass wir das self.calc.countChanged erstellte Signal self.calc.countChanged 
    mit der zum Aktualisieren des Fortschrittsbalkenwerts verwendeten Methode 
    verbunden haben. 
    Bei jeder Aktualisierung von External::run::count wird der int Wert 
    auch an onCountChanged gesendet. 
    """


    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Datei-Dialog und Textarea')
        self.setGeometry(0, 0, 640, 480)
        self.openFileNameDialog()
        self.openFileNamesDialog()
        self.saveFileDialog()        
        # feddich :)
        self.show()               

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
    
    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)
    
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
        
# ---------------------------------------------
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Actions()
    sys.exit(app.exec_())
