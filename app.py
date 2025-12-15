import io
import os
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pyopenms as oms
import torch.nn as nn
import tempfile  # <---【新增 1】必须引入这个库来处理云端文件
from jcamp import jcamp_readfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 必须包含的模型类定义 (保持不变)
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
        if len(total) == 0: return [] # 防止空数据报错
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
        p.setValue("tolerance", tolerance)
        p.setValue("is_relative_tolerance", "false")
        spa.setParameters(p)
        spa.getSpectrumAlignment(alignment, ref_spectrum, obs_spectrum)
        return alignment

    # 按参比文件_2
    def mass_calculation_ref(self,re_spectrum,ob_spectrum,alignment,decimal=4):
        ref = [i[0] for i in alignment]
        obs = [j[1] for j in alignment]
        for i,j in zip(ref,obs):
            # 增加越界检查，防止报错
            if i < len(re_spectrum) and j < len(ob_spectrum):
                ob_spectrum.iloc[j, 0] = re_spectrum.iloc[i, 0]
        return re_spectrum,ob_spectrum
    
    def load_imputer(self):
        # 你的预处理模型路径 (相对路径)
        scaler_path = r'imputer_scaler_model.pkl' 
        if not os.path.exists(scaler_path):
            st.error(f"❌ 找不到预处理文件：{scaler_path}")
            return None
        try:
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            st.error(f"预处理器加载出错: {e}")
            return None
    
    # 数据对齐和整合
    def prediction_pretreatment(self,uploaded_files,ages:list[int],genders:list[int]):
        # 上传参比数据
        sample_name = []
        # 确保 train_target.xlsx 在 GitHub 仓库里
        if not os.path.exists('train_target.xlsx'):
            st.error("❌ 找不到 train_target.xlsx")
            return None, None
            
        thyroid_train = pd.read_excel('train_target.xlsx')
        prim = thyroid_train.iloc[:,0:2]
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 数据对齐
        for i, file in enumerate(uploaded_files):
            # ---【新增 2】创建临时文件处理逻辑 ---
            # Streamlit 的 file 对象在内存中，jcamp 需要物理路径
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jdx") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # 读取临时文件路径
                jdxfile = jcamp_readfile(tmp_file_path) 
                
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
                prim = pd.merge(prim,o_spectrum,how='left',on='mass') 
                
            finally:
                # 必须删除临时文件
                os.remove(tmp_file_path)
            # -----------------------------------
        
        # 删去辅助数据
        prediction = prim.drop(prim.columns[1], axis=1).iloc[:,1:].T
        
        # 填充可能产生的 NaN (merge left 可能会产生空值)
        prediction = prediction.fillna(0)
        
        scaler = self.load_imputer()
        prediction = scaler.transform(prediction)
        
        # 年龄和性别数据整合
        buckets=[0,20,40,60,80]  
        age_bucket = np.digitize(ages, buckets, right=False) - 1
        
        # ---【新增 3】修复性别逻辑 ---
        # 你的 UI 传进来的是数字 1(男) 或 0(女)，不是字符串 'male'
        # 原代码：if m == 'male' (永远为 False)
        # 修改为：if m == 1
        categoricals = np.array([[0,1] if m == 1 else [1,0] for m in genders]).T
        
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
    model_path = r'PDD.pth'  # 保持相对路径，不要写 D:\
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
uploaded_files = st.file_uploader("可以上传多个JDX文件", type=["jdx", "dx"], accept_multiple_files=True)
ages = []
genders = []

if uploaded_files:
    # 加载年龄和性别数据
    for i, file in enumerate(uploaded_files):
        st.subheader(f"文件 {i+1}: {file.name}")
        age = st.number_input(f"输入年龄 ({file.name})", min_value=0, max_value=120, value=25, key=f"age_{i}")
        # 这里 1=男, 0=女
        gender = st.selectbox(f"选择性别 ({file.name})", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女", key=f"gender_{i}")
        ages.append(age)
        genders.append(gender)

    if st.button("开始训练并预测"):
        # 数据预处理
        prepossessor = Data_prepossing(SEQ_LENGTH=3,SEQ_SIZE=80)
        
        # 调用处理函数
        input_tensor, col_name = prepossessor.prediction_pretreatment(uploaded_files, ages, genders)
        
        if input_tensor is not None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 初始化加载
            model = load_deep_model()
            if model:
                model = model.to(DEVICE)

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
