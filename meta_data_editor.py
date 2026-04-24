import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QCompleter
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QStringListModel, QThread, pyqtSignal, QTimer
from mutagen.id3 import ID3, TIT2, ID3NoHeaderError
from google import genai
from google.genai import types
import config
from datetime import datetime

class AutoCompleteTrie:
   def __init__(self):
       self.root = {}

   def insert(self, word):
       node = self.root
       for char in word:
           if char not in node:
               node[char] = {}
           node = node[char]

   def find_suggestions(self, prefix):
       node = self.root
       suggestions = []
       for char in prefix:
           if char not in node:
               return []
           node = node[char]

       def traverse(node, word):
           if not node:
               suggestions.append(word)
           else:
               for char, child in node.items():
                   traverse(child, word + char)

       traverse(node, prefix)
       return suggestions

class SuggestionGeneratorThread(QThread):
   suggestions_generated = pyqtSignal(list)

   def __init__(self, prefix):
       super().__init__()
       self.prefix = prefix

   def run(self):
       try:
           client = genai.Client(api_key=config.GEMINI_API_KEY)
           prompt = (
               f"Suggest multiple emotive expressions in all lowercase and separated by commas "
               f"for the given prefix: {self.prefix.lower()}. "
               f"Example: laughs could be laughs out loud, laughs, laughs awkwardly, laughs nervously. "
               f"No numbering or commentary."
           )
           response = client.models.generate_content(
               model=config.GEMINI_TASK_MODEL_NAME,
               contents=prompt,
               config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=50),
           )
           text = response.text.strip() if response.text else ""
           suggestions = [s.strip() for s in text.split(",") if s.strip()]
           suggestions = list(set(suggestions))
           self.suggestions_generated.emit(suggestions)
       except Exception as e:
           print(f"Error generating model suggestions: {e}")
           self.suggestions_generated.emit([])

class AudioPlayer(QMainWindow):
   def __init__(self):
       super().__init__()
       self.setWindowTitle("Audio Meta Tag Editor")
       self.player = QMediaPlayer()
       self.suggestion_timer = None
       font = app.font()
       font.setPointSize(14)
       app.setFont(font)
       self.setStyleSheet("""
           QWidget {
               background-color: #2B2B2B;
               color: #D4D4D4;
           }
           QTreeWidget::item:selected {
               background-color: #505050;
               color: #D4D4D4;
           }
           QPushButton {
               background-color: #505050;
               color: #D4D4D4;
               padding: 5px;
               border-radius: 3px;
           }
           QPushButton:hover {
               background-color: #707070;
           }
           QPushButton:pressed {
               background-color: #808080;
           }
       """)

       open_button = QPushButton("Open Directory")
       open_button.clicked.connect(self.open_directory)

       self.file_tree = QTreeWidget()
       self.file_tree.setHeaderLabels(["File Name", "Meta Tag", "Date Modified"])
       self.file_tree.itemSelectionChanged.connect(self.play_audio)
       self.file_tree.setSelectionMode(QTreeWidget.SingleSelection)
       self.file_tree.setSortingEnabled(True)

       self.tag_edit = QLineEdit()
       self.tag_edit.editingFinished.connect(self.save_tag)
       self.tag_edit.returnPressed.connect(self.save_tag)
       self.tag_edit.textEdited.connect(self.update_suggestions)

       self.autocomplete_trie = AutoCompleteTrie()
       self.completer = QCompleter(self)
       self.completer.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
       self.completer.setCompletionMode(QCompleter.PopupCompletion)
       self.completer.activated.connect(self.save_tag)
       self.tag_edit.setCompleter(self.completer)

       layout = QVBoxLayout()
       layout.addWidget(open_button)
       layout.addWidget(self.file_tree)
       tag_layout = QHBoxLayout()
       tag_layout.addWidget(self.tag_edit)
       layout.addLayout(tag_layout)

       widget = QWidget()
       widget.setLayout(layout)
       self.setCentralWidget(widget)

       screen = app.primaryScreen().availableGeometry()
       self.setGeometry(
           screen.width() // 4,
           screen.height() // 4,
           screen.width() // 2,
           screen.height() // 2
       )

   def open_directory(self):
       default_path = os.path.join(os.getcwd(), "data", "audio")
       if not os.path.exists(default_path):
           default_path = os.getcwd()
           
       directory = QFileDialog.getExistingDirectory(self, "Open Directory", default_path)
       if directory:
           self.load_files(directory)

   def load_files(self, directory):
       self.file_tree.clear()
       self.autocomplete_trie = AutoCompleteTrie()
       for filename in os.listdir(directory):
           if filename.endswith(".mp3"):
               file_path = os.path.join(directory, filename)
               tag = self.get_tag(file_path)
               self.autocomplete_trie.insert(tag.lower())
               modified_time = os.path.getmtime(file_path)
               modified_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
               item = QTreeWidgetItem([filename, tag, modified_date])
               item.setData(0, Qt.UserRole, file_path)
               self.file_tree.addTopLevelItem(item)

   def get_tag(self, file_path):
       try:
           audio = ID3(file_path)
           if "TIT2" in audio:
               return audio["TIT2"].text[0]
       except:
           pass
       return "Unknown"

   def play_audio(self):
       item = self.file_tree.currentItem()
       if item:
           file_path = item.data(0, Qt.UserRole)
           if os.path.exists(file_path):
               self.player.stop()
               self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
               self.player.play()
               tag = item.text(1)
               self.tag_edit.setText(tag)
               self.tag_edit.setFocus()
               self.update_suggestions(tag)
               self.player.stateChanged.connect(self.on_player_state_changed)
           else:
               print(f"File not found: {file_path}")

   def update_suggestions(self, text):
       prefix = text.lower()
       self.trie_suggestions = self.autocomplete_trie.find_suggestions(prefix)

       if self.trie_suggestions:
           self.completer.setCompletionPrefix(prefix)
       else:
           self.completer.setCompletionPrefix("")
           self.completer.popup().hide()

       self.suggestion_generator_thread = SuggestionGeneratorThread(prefix)
       self.suggestion_generator_thread.suggestions_generated.connect(self.on_suggestions_generated)

       if self.suggestion_timer:
           self.suggestion_timer.stop()

       self.suggestion_timer = QTimer()
       self.suggestion_timer.setSingleShot(True)
       self.suggestion_timer.timeout.connect(self.suggestion_generator_thread.start)
       self.suggestion_timer.start(200)  # Delay in milliseconds (adjust as needed)

   def on_suggestions_generated(self, model_suggestions):
       prefix = self.tag_edit.text().lower()
       trie_suggestions = self.autocomplete_trie.find_suggestions(prefix)
       all_suggestions = trie_suggestions + [suggestion for suggestion in model_suggestions if not suggestion.lower().startswith(prefix)]
       model = QStringListModel(all_suggestions)
       self.completer.setModel(model)
       self.completer.setCompletionPrefix("")

       if all_suggestions:
           self.completer.complete()
       else:
           self.completer.popup().hide()

       print(f"Updated suggestions: {all_suggestions}")

   def on_player_state_changed(self, state):
       if state == QMediaPlayer.StoppedState:
           self.player.setMedia(QMediaContent())
           self.player.stateChanged.disconnect(self.on_player_state_changed)

   def save_tag(self):
       item = self.file_tree.currentItem()
       if item:
           file_path = item.data(0, Qt.UserRole)
           new_tag = self.tag_edit.text()
           try:
               audio = ID3(file_path)
               audio["TIT2"] = TIT2(encoding=3, text=new_tag)
               audio.save()
               item.setText(1, new_tag)
               self.autocomplete_trie.insert(new_tag.lower())
           except ID3NoHeaderError:
               # If the ID3 tag doesn't exist, create a new one
               audio = ID3()
               audio["TIT2"] = TIT2(encoding=3, text=new_tag)
               audio.save(file_path)
               item.setText(1, new_tag)
               self.autocomplete_trie.insert(new_tag.lower())
           except Exception as e:
               print(f"Failed to save tag for file: {file_path}")
               print(f"Error: {str(e)}")
       else:
           print("No file selected.")

   def keyPressEvent(self, event):
       if event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
           current_row = self.file_tree.currentIndex().row()
           if event.key() == Qt.Key_Up:
               new_row = max(0, current_row - 1)
           else:
               new_row = min(self.file_tree.topLevelItemCount() - 1, current_row + 1)
           self.file_tree.setCurrentIndex(self.file_tree.topLevelItem(new_row).index())
           self.play_audio()
       else:
           super().keyPressEvent(event)

if __name__ == "__main__":
   app = QApplication(sys.argv)
   player = AudioPlayer()
   player.show()
   sys.exit(app.exec_())