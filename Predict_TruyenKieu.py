import wx
import numpy as np
import collections
from tensorflow.keras.models import load_model 

def read_data(fname):
    with open(fname, encoding='utf-8') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    words = []
    for line in content:
        words.extend(line.split())
    return np.array(words)

def build_dataset(words):
    count = collections.Counter(words).most_common()
    word2id = {}
    for word, freq in count:
        word2id[word] = len(word2id)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return word2id, id2word

data = read_data('truyenkieu.txt')
w2i , i2w = build_dataset(data)
vocab_size = len(w2i)
timestep = 3

model = load_model("model.keras")

def MakePrediction(model, firstWords):
    sentence = firstWords.split()
    for _ in range(10) :
        encoded_input = np.array([[w2i[word] for word in sentence[-3:]]])
        y_pred = model.predict(encoded_input)
        pred_word = i2w[np.argmax(y_pred)]
        sentence.append(pred_word)
    return ' '.join(sentence)

class MyApp(wx.App):
    def OnInit(self):
        frame = wx.Frame(parent=None, title="Chào mừng  đến với wxPython", size=(400, 600))
        panel = wx.Panel(frame)

        label = wx.StaticText(panel, label="Tạo Đoạn Tiếp theo cho Truyện Kiều", pos=(40, 50))

        textCtrl = wx.TextCtrl(panel, pos=(50, 100), size=(300, 20), style=wx.TE_PROCESS_ENTER)
        textCtrl.Bind(wx.EVT_TEXT_ENTER, self.onEnterPressed)
        self.resultLabel = wx.StaticText(panel, label="", pos=(50, 150), size=(300, 200))

        frame.Show()
        return True

    def onEnterPressed(self, event):
        first_words = event.GetString()
        prediction = MakePrediction(model, first_words)
        self.resultLabel.SetLabel(f"{prediction}")

if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
