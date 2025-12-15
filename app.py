import io
import os
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pyopenms as oms
import torch.nn as nn
import tempfile  # <--- 必须保留这个
from jcamp import jcamp_readfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 模型类定义 (保持不变)
# ==========================================
class BiLSTM(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,num_classes,BiDirection=True):
        super(BiLSTM, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size) 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.BiDirection = BiDirection
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=0.3,batch_first=True,bidirectional=BiDirection)        
        if self.BiDirection == True:
            self.fc = nn.Linear(hidden_size*2,num_classes)
        else:
            self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        if self.BiDirection == True:
            hc0 = self.num_layers*2
        else:
            hc0 = self.num_layers
        h0 = torch.zeros(hc0, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(hc0, x.size(0), self.hidden_size).to(x.device)
        out, (h_n,c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. 预处理类 (仅修改必要部分)
# ==========================================
class Data_prepossing:
    def __init__(self,tolerance:float=0.5,SEQ_LENGTH:int=3,SEQ_SIZE:int=80):
        super(Data_prepossing, self).__init__()
        self.tolerance = tolerance
        self.seq_length = SEQ_LENGTH
        self.seq_size = SEQ_SIZE
    
    def noise_removal(self,mass_list,tolerance=0.5):
        total = mass_list.tolist()
        if len(total) == 0: return []
        ref_total = total[1:]+[[0,0]]
        new_total = [[r[0]-m[0],r[1]-m[1]] for r,m in zip(ref_total,total)]
        tf = [total[0]]
        for new,ref,to in zip(new_total,ref_total,total):
            if new[0] >= tolerance:
                tf = tf+[ref]
            else:
                if new[1]>=0:
                    tf = tf[:-1]+[ref]+[ref]
                else:
                    tf = tf[:-1]+[to]+[to]
        tf = [m for i,m in enumerate(tf) if m not in tf[:i]]
        return tf

    def openms_data_format(self,mass,intensity,decimal=5):
        mz = np.round(mass.values,decimal)
        mz_intensity = intensity.values
        spectrum = oms.MSSpectrum()
        spectrum.set_peaks([mz,mz_intensity])
        spectrum.sortByPosition()
        return spectrum

    def mass_align(self,ref_spectrum,obs_spectrum,tolerance=0.5):
        alignment = []
        spa = oms.SpectrumAlignment()
        p = spa.getParameters()
        p.setValue("tolerance", tolerance)
        p.setValue("is_relative_tolerance", "false")
        spa.setParameters(p)
        spa.getSpectrumAlignment(alignment, ref_spectrum, obs_spectrum)
        return alignment

    def mass_calculation_ref(self,re_spectrum,ob_spectrum,alignment,decimal=4):
        ref = [i[0] for i in alignment]
        obs = [j[1] for j in alignment]
        for i,j in zip(ref,obs):
            if i < len(re_spectrum) and j < len(ob_spectrum):
                ob_spectrum.iloc[j, 0] = re_spectrum.iloc[i, 0]
        return re_spectrum,ob_spectrum
    
    def load_imputer(self):
        # 相对路径
        scaler_path = 'imputer_scaler_model.pkl' 
        if not os.path.exists(scaler_path):
            st.error(f"❌ 找不到预处理文件：{scaler_path}")
            return None
        try:
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            st.error(f"预处理器加载出错: {e}")
            return None
    
    def prediction_pretreatment(self,uploaded_files,ages:list[int],genders:list[int]):
        sample_name = []
        if not os.path.exists('train_target.xlsx'):
            st.error("❌ 找不到 train_target.xlsx")
            return None, None
            
        thyroid_train = pd.read_excel('train_target.xlsx')
        prim = thyroid_train.iloc[:,0:2]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        for i, file in enumerate(uploaded_files):
            # 临时文件处理
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jdx") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                jdxfile = jcamp_readfile(tmp_file_path) 
                indata = np.vstack((jdxfile['x'],jdxfile['y'])).T
                denoise = self.noise_removal(indata,tolerance=self.tolerance)
                framefile = pd.DataFrame(denoise,columns=['mass',file.name])
                sample_name.append(file.name)
                
                ref_spectrum = self.openms_data_format(prim.mass,prim.iloc[:,1])
                obs_spectrum = self.openms_data_format(framefile.mass,framefile.iloc[:,1])
                alignment = self.mass_align(ref_spectrum,obs_spectrum,tolerance=self.tolerance)
                
                r_spectrum,o_spectrum = self.mass_calculation_ref(prim,framefile,alignment)
                prim = pd.merge(prim,o_spectrum,how='left',on='mass') 
            finally:
                os.remove(tmp_file_path)
        
        # 删去辅助数据
        prediction = prim.drop(prim.columns[1], axis=1).iloc[:,1:].T
        prediction = prediction.fillna(0)
        
        # ========================================================
        # 🛠️ 关键修复：强制转换为 float 类型
        # ========================================================
        # 这可以防止新版 sklearn 在处理 object 类型时触发 _fill_dtype 错误
        prediction = prediction.astype(float) 

        scaler = self.load_imputer()
        prediction = scaler.transform(prediction)
        
        buckets=[0,20,40,60,80]  
        age_bucket = np.digitize(ages, buckets, right=False) - 1
        categoricals = np.array([[0,1] if m == 1 else [1,0] for m in genders]).T
        totals = np.vstack((prediction.T,age_bucket.astype('float32'),categoricals))
        
        prediction_tensor = torch.tensor(totals.T,dtype=torch.float32)
        prediction_seq = prediction_tensor.view(-1,self.seq_length,self.seq_size).to(DEVICE)
        
        return prediction_seq,sample_name

# ==========================================
# 3. Streamlit 界面 (保持不变)
# ==========================================
@st.cache_resource
def load_deep_model():
    model_path = 'PDD.pth' 
    if not os.path.exists(model_path):
        st.error(f"❌ 找不到 LSTM 模型文件：{model_path}")
        return None
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"LSTM 模型加载出错: {e}")
        return None

st.title("基于DPiMS和深度学习的牙周病诊断系统")
uploaded_files = st.file_uploader("可以上传多个JDX文件", type=["jdx", "dx"], accept_multiple_files=True)
ages = []
genders = []

if uploaded_files:
    for i, file in enumerate(uploaded_files):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: st.text(f"📁 {file.name}")
        with c2: 
            age = st.number_input(f"年龄", 0, 120, 25, key=f"age_{i}", label_visibility="collapsed")
            ages.append(age)
        with c3:
            gender = st.selectbox(f"性别", [1, 0], format_func=lambda x: "男" if x==1 else "女", key=f"gender_{i}", label_visibility="collapsed")
            genders.append(gender)

    if st.button("开始诊断"):
        prepossessor = Data_prepossing(SEQ_LENGTH=3,SEQ_SIZE=80)
        input_tensor,col_name = prepossessor.prediction_pretreatment(uploaded_files,ages,genders)
        
        if input_tensor is not None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_deep_model()
            if model:
                model = model.to(DEVICE)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)
                    confidence = torch.max(probabilities, dim=1)[0].cpu()

                st.divider()
                st.subheader("🔮 诊断结果")
                group_name = ['健康','牙周炎+糖尿病','牙周炎']
                results = []
                for col,m,n in zip(col_name,predicted_class,confidence):
                    results.append({"样本名称": col, "诊断结果": group_name[m], "置信度": f"{n.item():.2%}"})
                st.table(results)
