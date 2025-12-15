import io
import os
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pyopenms as oms
import torch.nn as nn
from jcamp import jcamp_readfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 必须包含的模型类定义
# ==========================================
class BiLSTM(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,num_classes,BiDirection=True):
        super(BiLSTM, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size) 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.BiDirection = BiDirection
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=0.3,batch_first=True,bidirectional=BiDirection)        
        # 全连接输出层
        if self.BiDirection == True:
            self.fc = nn.Linear(hidden_size*2,num_classes)
        else:
            self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        if self.BiDirection == True:
            hc0 = self.num_layers*2
        else:
            hc0 = self.num_layers
        h0 = torch.zeros(hc0, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(hc0, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, (h_n,c_n) = self.lstm(x, (h0, c0))
        
        # 全连接层预测
        out = self.fc(out[:, -1, :])
        
        return out

# ==========================================
# 2. 模型预处理类定义
# ==========================================
class Data_prepossing:
    def __init__(self,tolerance:float=0.5,SEQ_LENGTH:int=3,SEQ_SIZE:int=80):
        super(Data_prepossing, self).__init__()
        self.tolerance = tolerance
        self.seq_length = SEQ_LENGTH
        self.seq_size = SEQ_SIZE
    
    # 数据质控，质谱峰去噪，去平头峰
    def noise_removal(self,mass_list,tolerance=0.5):
        total = mass_list.tolist()
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

    # 生成openms数据格式
    def openms_data_format(self,mass,intensity,decimal=5):
        # 质谱保留
        mz = np.round(mass.values,decimal)
        mz_intensity = intensity.values
        spectrum = oms.MSSpectrum()
        spectrum.set_peaks([mz,mz_intensity])
        spectrum.sortByPosition()
        return spectrum

    # 质量数对齐
    def mass_align(self,ref_spectrum,obs_spectrum,tolerance=0.5):
        alignment = []
        spa = oms.SpectrumAlignment()
        p = spa.getParameters()
        # use 0.5 Da tolerance (Note: for high-resolution data we could also 
        # use ppm by setting the is_relative_tolerance value to true)
        p.setValue("tolerance", tolerance)
        p.setValue("is_relative_tolerance", "false")
        spa.setParameters(p)
        # align both spectra
        spa.getSpectrumAlignment(alignment, ref_spectrum, obs_spectrum)
        return alignment

    # 按参比文件_2
    def mass_calculation_ref(self,re_spectrum,ob_spectrum,alignment,decimal=4):
        ref = [i[0] for i in alignment]
        obs = [j[1] for j in alignment]
        for i,j in zip(ref,obs):
            ob_spectrum.iloc[j, 0] = re_spectrum.iloc[i, 0]
        return re_spectrum,ob_spectrum
    
    def load_imputer(self):
        # 你的预处理模型路径
        scaler_path = r'imputer_scaler_model.pkl' 
        if not os.path.exists(scaler_path):
            st.error(f"❌ 找不到预处理文件：{scaler_path}")
            return None
        try:
            # 使用 joblib 加载 sklearn 的对象
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            st.error(f"预处理器加载出错: {e}")
            return None
    
    # 数据对齐和整合
    def prediction_pretreatment(self,uploaded_file:list[str],ages:list[int],genders:list[str]):
        # 上传参比数据
        sample_name = []
        thyroid_train = pd.read_excel('train_target.xlsx')
        prim = thyroid_train.iloc[:,0:2]
        #col_list = [a.split('\\')[-1].split('.')[0] for a in uploaded_file]
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 数据对齐
        for i,file in enumerate(uploaded_file):
            # 读取文件
            jdxfile = jcamp_readfile(file.name) 
            indata = np.vstack((jdxfile['x'],jdxfile['y'])).T
            denoise = self.noise_removal(indata,tolerance=self.tolerance) # 去除噪声
            framefile = pd.DataFrame(denoise,columns=['mass',file.name])
            sample_name.append(file.name)
            
            # 数据验证
            ref_spectrum = self.openms_data_format(prim.mass,prim.iloc[:,1])
            obs_spectrum = self.openms_data_format(framefile.mass,framefile.iloc[:,1])
            alignment = self.mass_align(ref_spectrum,obs_spectrum,tolerance=self.tolerance)
            
            # 数据整合
            r_spectrum,o_spectrum = self.mass_calculation_ref(prim,framefile,alignment)
            prim = pd.merge(prim,o_spectrum,how='left',on='mass') # merge用好太不容易了
        
        # 删去辅助数据
        prediction = prim.drop(prim.columns[1], axis=1).iloc[:,1:].T
        scaler = self.load_imputer()
        prediction = scaler.transform(prediction)
        
        # 年龄和性别数据整合
        buckets=[0,20,40,60,80]  
        age_bucket = np.digitize(ages, buckets, right=False) - 1
        categoricals = np.array([[0,1] if m == 'male' else [1,0] for m in genders]).T
        totals = np.vstack((prediction.T,age_bucket.astype('float32'),categoricals))
        
        # 转换为tensor，并在Device上运行
        prediction_tensor = torch.tensor(totals.T,dtype=torch.float32)
        prediction_seq = prediction_tensor.view(-1,self.seq_length,self.seq_size).to(DEVICE)
        
        return prediction_seq,sample_name

# ==========================================
# 3. stremlit界面及模型上载
# ==========================================
@st.cache_resource
def load_deep_model():
    model_path = r'PDD.pth'  # 你的 PyTorch 模型路径
    if not os.path.exists(model_path):
        st.error(f"❌ 找不到 LSTM 模型文件：{model_path}")
        return None
    try:
        # weights_only=False 解决版本兼容性报错
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"LSTM 模型加载出错: {e}")
        return None

# Streamlit界面
st.title("基于DPiMS和深度学习的牙周病诊断系统")
uploaded_files = st.file_uploader("可以上传多个JDX文件", type="jdx", accept_multiple_files=True)
ages = []
genders = []

if uploaded_files:
    # 加载年龄和性别数据
    for i, file in enumerate(uploaded_files):
        st.subheader(f"文件 {i+1}: {file.name}")
        age = st.number_input(f"输入年龄 ({file.name})", min_value=0, max_value=120, value=25, key=f"age_{i}")
        gender = st.selectbox(f"选择性别 ({file.name})", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女", key=f"gender_{i}")
        ages.append(age)
        genders.append(gender)

    if st.button("开始训练并预测"):
        # 数据预处理
        prepossessor = Data_prepossing(SEQ_LENGTH=3,SEQ_SIZE=80)
        input_tensor,col_name = prepossessor.prediction_pretreatment(uploaded_files,ages,genders)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化加载
        model = load_deep_model().to(DEVICE)

        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0].cpu()

        # ==========================================
        # 5. 结果展示
        # ==========================================
        st.divider()
        st.subheader("🔮 分析报告")
        group_name = ['健康','牙周炎+糖尿病','牙周炎']
        for col,m,n in zip(col_name,predicted_class,confidence):
            col1, col2, col3 = st.columns(3)
            col1.metric("样品名称", f"{col}")
            col2.metric("预测类别", f"{group_name[m]}")
            col3.metric("置信度", f"{n.item():.1%}")

            # 用颜色条展示置信度
            #if confidence > 0.8:
            #    st.success("模型对此结果非常有信心。")
            #elif confidence > 0.5:
            #    st.warning("模型信心一般，建议结合临床判断。")
            #else:
            #    st.error("模型信心不足，结果仅供参考。")
    else:
        st.warning("⚠️ 请检查 文件夹下是否同时存在 PDD.pth 和 imputer_scaler_model.pkl")