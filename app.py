import io
import os
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pyopenms as oms
import torch.nn as nn
# 引入 tempfile 处理上传文件
import tempfile 
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
        if len(total) < 2: return total # 防止数据过少报错
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
        # 注意：这里需要确保索引不越界
        for i,j in zip(ref,obs):
            if i < len(re_spectrum) and j < len(ob_spectrum):
                ob_spectrum.iloc[j, 0] = re_spectrum.iloc[i, 0]
        return re_spectrum,ob_spectrum
    
    def load_imputer(self):
        # 修改为相对路径
        scaler_path = 'imputer_scaler_model.pkl' 
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
    # uploaded_file 实际上是 UploadedFile 对象列表，不是字符串列表
    def prediction_pretreatment(self,uploaded_files,ages:list[int],genders:list[int]):
        sample_name = []
        
        # 读取同级目录下的 excel
        if not os.path.exists('train_target.xlsx'):
            st.error("❌ 找不到 train_target.xlsx，请确保已上传到 GitHub/云端")
            return None, None
            
        thyroid_train = pd.read_excel('train_target.xlsx')
        prim = thyroid_train.iloc[:,0:2]
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 数据对齐
        for i, file in enumerate(uploaded_files):
            # ==============================================
            # 🛠️ 关键修复：使用 tempfile 处理内存文件
            # ==============================================
            file_name = file.name
            sample_name.append(file_name)
            
            # 创建临时文件
            suffix = ".jdx" if not file_name.endswith(".jdx") else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix + ".jdx") as tmp_file:
                tmp_file.write(file.getvalue()) # 写入数据
                tmp_file_path = tmp_file.name

            try:
                # 读取临时文件
                jdxfile = jcamp_readfile(tmp_file_path)
                indata = np.vstack((jdxfile['x'],jdxfile['y'])).T
                
                # 去除噪声
                denoise = self.noise_removal(indata,tolerance=self.tolerance) 
                framefile = pd.DataFrame(denoise,columns=['mass',file_name])
                
                # 数据验证
                ref_spectrum = self.openms_data_format(prim.mass,prim.iloc[:,1])
                obs_spectrum = self.openms_data_format(framefile.mass,framefile.iloc[:,1])
                alignment = self.mass_align(ref_spectrum,obs_spectrum,tolerance=self.tolerance)
                
                # 数据整合
                r_spectrum,o_spectrum = self.mass_calculation_ref(prim,framefile,alignment)
                # left merge 保证保留 reference 的 mass
                prim = pd.merge(prim, o_spectrum, how='left', on='mass') 
                
            except Exception as e:
                st.error(f"处理文件 {file_name} 时出错: {e}")
                st.stop()
            finally:
                # 删除临时文件
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
        
        # 删去辅助数据 (去掉前两列：mass 和 train_target 的 intensity)
        # 注意：prim 的结构变成了 [mass, ref_intensity, file1, file2...]
        # 所以要 drop 前两列
        try:
            prediction = prim.iloc[:, 2:].T # 转置后：行是样本，列是特征
            
            # 检查特征数量是否匹配 scaler
            scaler = self.load_imputer()
            if scaler:
                # 如果有缺失值（因为 merge left 可能产生 NaN），先填充 0
                prediction = prediction.fillna(0)
                prediction = scaler.transform(prediction)
            
            # 年龄和性别数据整合
            buckets=[0,20,40,60,80]  
            age_bucket = np.digitize(ages, buckets, right=False) - 1
            
            # 🛠️ 关键修复：性别逻辑
            # UI 输入：1 (男), 0 (女)
            # 原逻辑：'male' -> [0, 1]
            # 新逻辑：1 -> [0, 1], 0 -> [1, 0]
            categoricals = np.array([[0,1] if m == 1 else [1,0] for m in genders]).T
            
            totals = np.vstack((prediction.T, age_bucket.astype('float32'), categoricals))
            
            # 转换为tensor
            prediction_tensor = torch.tensor(totals.T, dtype=torch.float32)
            
            # 检查维度是否匹配 reshape
            expected_features = self.seq_length * self.seq_size
            if prediction_tensor.shape[1] != expected_features:
                 st.error(f"维度错误：处理后特征数为 {prediction_tensor.shape[1]}，但模型需要 {expected_features}")
                 st.info("提示：请检查 train_target.xlsx 的行数或预处理逻辑是否与训练时一致。")
                 st.stop()

            prediction_seq = prediction_tensor.view(-1, self.seq_length, self.seq_size).to(DEVICE)
            
            return prediction_seq, sample_name
            
        except Exception as e:
            st.error(f"数据整合阶段出错: {e}")
            return None, None

# ==========================================
# 3. Streamlit 界面及模型上载
# ==========================================
@st.cache_resource
def load_deep_model():
    # 相对路径
    model_path = 'PDD.pth'  
    if not os.path.exists(model_path):
        return None
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"LSTM 模型加载出错: {e}")
        return None

# 初始化
st.title("基于DPiMS和深度学习的牙周病诊断系统")

# 检查模型是否存在
model = load_deep_model()
if model is None:
    st.error("❌ 找不到 PDD.pth，请确保文件已上传！")
    st.stop()

uploaded_files = st.file_uploader("请上传 JDX 质谱文件", type=["jdx", "dx"], accept_multiple_files=True)

ages = []
genders = []

if uploaded_files:
    st.divider()
    st.write("### 1. 填写患者信息")
    # 使用 columns 布局更紧凑
    for i, file in enumerate(uploaded_files):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.text(f"📁 {file.name}")
        with c2:
            age = st.number_input(f"年龄", min_value=0, max_value=120, value=25, key=f"age_{i}", label_visibility="collapsed")
            ages.append(age)
        with c3:
            # 这里 1=男, 0=女
            gender = st.selectbox(f"性别", options=[1, 0], format_func=lambda x: "男" if x == 1 else "女", key=f"gender_{i}", label_visibility="collapsed")
            genders.append(gender)
    
    st.divider()
    if st.button("开始诊断"):
        with st.spinner("正在进行数据预处理和分析..."):
            # 数据预处理
            # 确保 SEQ_SIZE 和 SEQ_LENGTH 与你训练时完全一致！
            prepossessor = Data_prepossing(SEQ_LENGTH=3, SEQ_SIZE=80)
            
            input_tensor, col_name = prepossessor.prediction_pretreatment(uploaded_files, ages, genders)
            
            if input_tensor is not None:
                DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(DEVICE)

                # 预测
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)
                    confidence = torch.max(probabilities, dim=1)[0].cpu()

                # 结果展示
                st.subheader("🔮 诊断结果")
                group_name = ['健康', '牙周炎+糖尿病', '牙周炎']
                
                # 创建结果表格
                results = []
                for col, m, n in zip(col_name, predicted_class, confidence):
                    res_dict = {
                        "样本名称": col,
                        "诊断结果": group_name[m],
                        "置信度": f"{n.item():.2%}"
                    }
                    results.append(res_dict)
                
                st.table(results)
