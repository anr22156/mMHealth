#!/usr/bin/env python
# coding: utf-8

# In[1]:


def ECGPreprocessing(data):
    import numpy as np
    from sklearn.preprocessing import minmax_scale
    from scipy.signal import find_peaks
    
    ECG_norm = minmax_scale(data,feature_range=(0, 1))
    
    peak = find_peaks(ECG_norm, height=0.01, prominence=0.01)
    r_peak = []
    p_peak = []
    t_peak = []
    for i in range(len(peak[0])-2):
        if i == 0:
            p_peak.append(peak[0][i])
        else:
            if ECG_norm[peak[0][i]] > 0.7:
                r_peak.append(peak[0][i])
                if ECG_norm[peak[0][i+1]] < 0.7:
                    p_peak.append(peak[0][i+2])
                if ECG_norm[peak[0][i+2]] < 0.7:
                    t_peak.append(peak[0][i+1])
    all_peaks = sorted([*set(r_peak + p_peak + t_peak)])
    
    q_pt = []
    s_pt = []
    q_idx = []
    s_idx = []
    for i in range(len(all_peaks)-1):
        if ECG_norm[all_peaks[i]] > 0.7:
            ptp1 = ECG_norm[all_peaks[i-1]:all_peaks[i]]
            if len(ptp1) != 0: 
                q_pt.append(min(ptp1))
                q_idx.append((np.where(ptp1 == min(ptp1))[0][0]+all_peaks[i-1]))
            ptp2 = ECG_norm[all_peaks[i]:all_peaks[i+1]]
            if len(ptp2) != 0:
                s_pt.append(min(ptp2))
                s_idx.append((np.where(ptp2 == min(ptp2))[0][0]+all_peaks[i]))
    
    ECG_features = [ECG_norm, p_peak, r_peak, t_peak, q_idx, q_pt, s_idx, s_pt]
    return ECG_features

def ECGPreprocessing1(data):
    import numpy as np
    from sklearn.preprocessing import minmax_scale
    from scipy.signal import find_peaks
    
    ECG_norm = minmax_scale(data,feature_range=(0, 1))
    
    peak = find_peaks(ECG_norm, height=0.01, prominence=0.01)
    r_peak = []
    p_peak = []
    t_peak = []
    for i in range(len(peak[0])-2):
        if ECG_norm[peak[0][i]] > 0.4:
            r_peak.append(peak[0][i])
            if ECG_norm[peak[0][i+1]] < 0.4:
                p_peak.append(peak[0][i+2])
            if ECG_norm[peak[0][i+2]] < 0.4:
                t_peak.append(peak[0][i+1])
    all_peaks = sorted([*set(r_peak + p_peak + t_peak)])
    
    q_pt = []
    s_pt = []
    q_idx = []
    s_idx = []
    for i in range(len(all_peaks)-1):
        if ECG_norm[all_peaks[i]] > 0.4:
            ptp1 = ECG_norm[all_peaks[i-1]:all_peaks[i]]
            if len(ptp1) != 0: 
                q_pt.append(min(ptp1))
                q_idx.append((np.where(ptp1 == min(ptp1))[0][0]+all_peaks[i-1]))
            ptp2 = ECG_norm[all_peaks[i]:all_peaks[i+1]]
            if len(ptp2) != 0:
                s_pt.append(min(ptp2))
                s_idx.append((np.where(ptp2 == min(ptp2))[0][0]+all_peaks[i]))
    
    ECG_features = [ECG_norm, p_peak, r_peak, t_peak, q_idx, q_pt, s_idx, s_pt]
    return ECG_features

# In[2]:


def PPGPreprocessing(data):
    import numpy as np
    import heartpy as hp
    from scipy.signal import find_peaks, argrelextrema
    from sklearn.preprocessing import minmax_scale
    
    PPG_norm = minmax_scale(data,feature_range=(0, 1))
    
    # 1st derivative
    PPG_d = np.gradient(PPG_norm)
    # 2nd derivative
    PPG_sd = np.gradient(PPG_d)
    
    # running the PPG peak detection analysis 
    wd1, m1 = hp.process(PPG_norm, sample_rate = 125)
    wd_d1, m_d1 = hp.process(PPG_d, sample_rate = 125)
    wd_sd1, m_sd1 = hp.process(PPG_sd, sample_rate = 125)
    
    # (a) systolic peaks
    systolic_peak_idx = wd1['peaklist'] # array of peak indices
    systolic_peak = wd1['ybeat'] # array of peak values
    
    # (b1) valley points for PPG
    ppg_valley_point_idx = [] # array of valley point indices
    ppg_valley_point = [] # array of valley point values
    for i in range(len(systolic_peak)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = systolic_peak_idx[i] # index of 1st peak
            ptp = PPG_norm[p1:p2]
            min_val = min(ptp)
            ppg_valley_point.append(min_val)
            min_idx = np.where(ptp == min_val)
            ppg_valley_point_idx.append(min_idx[0][0].tolist()) 
        else: 
            p1 = systolic_peak_idx[i-1] # index of 1st peak
            p2 = systolic_peak_idx[i] # index of 2nd peak
            ptp = PPG_norm[p1:p2]
            min_val = min(ptp)
            ppg_valley_point.append(min_val)
            min_idx = np.where(ptp == min_val) + p1
            ppg_valley_point_idx.append(min_idx[0][0].tolist())
            
    # (b2) valley points and lowest points for dPPG
    dppg_peak_idx = wd_d1['peaklist']
    dppg_peak = wd_d1['ybeat']
    dppg_valley_point_idx = [] # array of valley point indices
    dppg_valley_point = [] # array of valley point values
    dDA_point_idx = [] # array of lowest point indices
    dDA_point = [] # array of lowest point value

    # valley points
    for i in range(len(dppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            mins = argrelextrema(ptp, np.less_equal)
            dppg_valley_point_idx.append(mins[0][-1])
            dppg_valley_point.append(ptp[mins[0][-1]])
        else:
            p1 = dppg_peak_idx[i-1] # index of 1st peak
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            if len(ptp) > 20:
                mins = argrelextrema(ptp, np.less_equal)
                if ptp[mins[0][-1]] != 0:
                    dppg_valley_point_idx.append(mins[0][-1] + p1)
                    dppg_valley_point.append(ptp[mins[0][-1]])
            
    # lowest points
    for i in range(len(dppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            min_val = min(ptp)
            dDA_point.append(min_val)
            min_idx = np.where(ptp == min_val)
            dDA_point_idx.append(min_idx[0][0].tolist())
        else:
            p1 = dppg_peak_idx[i-1] # index of 1st peak
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            min_val = min(ptp)
            dDA_point.append(min_val)
            min_idx = np.where(ptp == min_val) + p1
            dDA_point_idx.append(min_idx[0][0].tolist())
            
    # (b3) valley points and lowest points for sdPPG
    sdppg_peak_idx = wd_sd1['peaklist']
    sdppg_peak = wd_sd1['ybeat']
    sdppg_valley_point_idx = [] # array of valley point indices
    sdppg_valley_point = [] # array of valley point values
    sdDA_point_idx = [] # array of lowest point indices
    sdDA_point = [] # array of lowest point value

    # valley points
    for i in range(len(sdppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            mins = argrelextrema(ptp, np.less_equal)
            if mins == True:
                sdppg_valley_point_idx.append(mins[0][-1])
                sdppg_valley_point.append(ptp[mins[0][-1]])
            else:
                sdppg_valley_point_idx.append(np.nan)
                sdppg_valley_point.append(np.nan)
        else:
            p1 = sdppg_peak_idx[i-1] # index of 1st peak
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            if len(ptp) > 20:
                mins = argrelextrema(ptp, np.less_equal)
                if ptp[mins[0][-1]] != 0:
                    sdppg_valley_point_idx.append(mins[0][-1] + p1)
                    sdppg_valley_point.append(ptp[mins[0][-1]])
            
    # lowest points
    for i in range(len(sdppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            min_val = min(ptp)
            sdDA_point.append(min_val)
            min_idx = np.where(ptp == min_val)
            sdDA_point_idx.append(min_idx[0][0].tolist())
        else:
            p1 = sdppg_peak_idx[i-1] # index of 1st peak
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            min_val = min(ptp)
            sdDA_point.append(min_val)
            min_idx = np.where(ptp == min_val) + p1
            sdDA_point_idx.append(min_idx[0][0].tolist())
            
     # (c) max/min slope values 
    slopeval_ppg = [PPG_norm[i] for i in wd_d1['peaklist']]
    slopeval_ppg_idx = wd_d1['peaklist']
    slopeval_dppg = [PPG_d[i] for i in wd_sd1['peaklist']]
    slopeval_dppg_idx = wd_sd1['peaklist']
    
    # (d) dicrotic notch points
    dn = find_peaks(PPG_sd, distance=25)
    secondary_idx = [dn[0][index] for index in range(0, len(dn[0]), 2)]
    dicrotic_notch_idx = [item for item in secondary_idx if PPG_sd[item] <= 0.05]
    dicrotic_notch = [PPG_norm[i] for i in dicrotic_notch_idx]
    
    PPG_features = [ppg_valley_point_idx,ppg_valley_point,dppg_valley_point_idx,dppg_valley_point,sdppg_valley_point_idx,sdppg_valley_point,systolic_peak_idx,systolic_peak,slopeval_ppg_idx,slopeval_ppg,slopeval_dppg_idx,slopeval_dppg,dicrotic_notch_idx,dicrotic_notch,dDA_point_idx,dDA_point,sdDA_point_idx,sdDA_point,dppg_peak_idx,dppg_peak,sdppg_peak_idx,sdppg_peak]
    return PPG_features, PPG_norm



def PPGPreprocessing1(data):
    import numpy as np
    import heartpy as hp
    from scipy.signal import find_peaks, argrelextrema
    from sklearn.preprocessing import minmax_scale
    
    PPG_norm = minmax_scale(data,feature_range=(0, 1))
    
    # 1st derivative
    PPG_d = np.gradient(PPG_norm)
    # 2nd derivative
    PPG_sd = np.gradient(PPG_d)
    
    # running the PPG peak detection analysis 
    wd1, m1 = hp.process(PPG_norm, sample_rate = 800)
    #wd_d1, m_d1 = hp.process(PPG_d, sample_rate = 800)
    #wd_sd1, m_sd1 = hp.process(PPG_sd, sample_rate = 800)
    
    # (a) systolic peaks
    systolic_peak_idx = wd1['peaklist'] # array of peak indices
    systolic_peak = wd1['ybeat'] # array of peak values
    
    # (b1) valley points for PPG
    ppg_valley_point_idx = [] # array of valley point indices
    ppg_valley_point = [] # array of valley point values
    for i in range(len(systolic_peak)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = systolic_peak_idx[i] # index of 1st peak
            ptp = PPG_norm[p1:p2]
            min_val = min(ptp)
            ppg_valley_point.append(min_val)
            min_idx = np.where(ptp == min_val)
            ppg_valley_point_idx.append(min_idx[0][0].tolist()) 
        else: 
            p1 = systolic_peak_idx[i-1] # index of 1st peak
            p2 = systolic_peak_idx[i] # index of 2nd peak
            ptp = PPG_norm[p1:p2]
            min_val = min(ptp)
            ppg_valley_point.append(min_val)
            min_idx = np.where(ptp == min_val) + p1
            ppg_valley_point_idx.append(min_idx[0][0].tolist())
            
    # (b2) valley points and lowest points for dPPG
    #dppg_peak_idx = wd_d1['peaklist']
    #dppg_peak = wd_d1['ybeat']
    peaks, _ = find_peaks(PPG_d, distance=500)
    dppg_peak_idx = peaks
    dppg_peak = PPG_d[peaks]
    dppg_valley_point_idx = [] # array of valley point indices
    dppg_valley_point = [] # array of valley point values
    dDA_point_idx = [] # array of lowest point indices
    dDA_point = [] # array of lowest point value

    # valley points
    for i in range(len(dppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            mins = argrelextrema(ptp, np.less_equal)
            dppg_valley_point_idx.append(mins[0][-1])
            dppg_valley_point.append(ptp[mins[0][-1]])
        else:
            p1 = dppg_peak_idx[i-1] # index of 1st peak
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            if len(ptp) > 20:
                mins = argrelextrema(ptp, np.less_equal)
                if ptp[mins[0][-1]] != 0:
                    dppg_valley_point_idx.append(mins[0][-1] + p1)
                    dppg_valley_point.append(ptp[mins[0][-1]])
            
    # lowest points
    for i in range(len(dppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            min_val = min(ptp)
            dDA_point.append(min_val)
            min_idx = np.where(ptp == min_val)
            dDA_point_idx.append(min_idx[0][0].tolist())
        else:
            p1 = dppg_peak_idx[i-1] # index of 1st peak
            p2 = dppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_d[p1:p2]
            min_val = min(ptp)
            dDA_point.append(min_val)
            min_idx = np.where(ptp == min_val) + p1
            dDA_point_idx.append(min_idx[0][0].tolist())
            
    # (b3) valley points and lowest points for sdPPG
    #sdppg_peak_idx = wd_sd1['peaklist']
    #sdppg_peak = wd_sd1['ybeat']
    peaks, _ = find_peaks(PPG_sd, distance=500)
    sdppg_peak_idx = peaks
    sdppg_peak = PPG_sd[sdppg_peak_idx]
    sdppg_valley_point_idx = [] # array of valley point indices
    sdppg_valley_point = [] # array of valley point values
    sdDA_point_idx = [] # array of lowest point indices
    sdDA_point = [] # array of lowest point value

    # valley points
    for i in range(len(sdppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            mins = argrelextrema(ptp, np.less_equal)
            if mins == True:
                sdppg_valley_point_idx.append(mins[0][-1])
                sdppg_valley_point.append(ptp[mins[0][-1]])
            else:
                sdppg_valley_point_idx.append(np.nan)
                sdppg_valley_point.append(np.nan)
        else:
            p1 = sdppg_peak_idx[i-1] # index of 1st peak
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            if len(ptp) > 20:
                mins = argrelextrema(ptp, np.less_equal)
                if ptp[mins[0][-1]] != 0:
                    sdppg_valley_point_idx.append(mins[0][-1] + p1)
                    sdppg_valley_point.append(ptp[mins[0][-1]])
            
    # lowest points
    for i in range(len(sdppg_peak_idx)-1):
        if i == 0:
            p1 = 0 # beginning of data
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            min_val = min(ptp)
            sdDA_point.append(min_val)
            min_idx = np.where(ptp == min_val)
            sdDA_point_idx.append(min_idx[0][0].tolist())
        else:
            p1 = sdppg_peak_idx[i-1] # index of 1st peak
            p2 = sdppg_peak_idx[i] # index of 2nd peak
            ptp = PPG_sd[p1:p2]
            min_val = min(ptp)
            sdDA_point.append(min_val)
            min_idx = np.where(ptp == min_val) + p1
            sdDA_point_idx.append(min_idx[0][0].tolist())
            
     # (c) max/min slope values 
    slopeval_ppg = [PPG_norm[i] for i in dppg_peak_idx]
    slopeval_ppg_idx = dppg_peak_idx
    slopeval_dppg = [PPG_d[i] for i in sdppg_peak_idx]
    slopeval_dppg_idx = sdppg_peak_idx
    
    # (d) dicrotic notch points
    dn = find_peaks(PPG_sd, distance=25)
    secondary_idx = [dn[0][index] for index in range(0, len(dn[0]), 2)]
    dicrotic_notch_idx = [item for item in secondary_idx if PPG_sd[item] <= 0.05]
    dicrotic_notch = [PPG_norm[i] for i in dicrotic_notch_idx]
    
    PPG_features = [ppg_valley_point_idx,ppg_valley_point,dppg_valley_point_idx,dppg_valley_point,sdppg_valley_point_idx,sdppg_valley_point,systolic_peak_idx,systolic_peak,slopeval_ppg_idx,slopeval_ppg,slopeval_dppg_idx,slopeval_dppg,dicrotic_notch_idx,dicrotic_notch,dDA_point_idx,dDA_point,sdDA_point_idx,sdDA_point,dppg_peak_idx,dppg_peak,sdppg_peak_idx,sdppg_peak]
    return PPG_features, PPG_norm

# In[3]:


def BPPreprocessing(data):
    import numpy as np
    from scipy.signal import find_peaks
    
    sys_idx, sys_val = find_peaks(data,height=(100,160))
    sys_val = sys_val.get('peak_heights')
    # lowest points
    dias_idx = []
    dias_val = []
    for i in range(len(sys_idx)-1):
        p1 = sys_idx[i] # index of 1st peak
        p2 = sys_idx[i+1] # index of 2nd peak
        ptp = data[p1:p2]
        min_val = min(ptp)
        dias_val.append(min_val)
        min_idx = min([np.abs(float(x)-float(min_val)) for x in ptp]) + p1
        #min_idx = np.abs(ptp - float(min_val)).argmin() + p1
        #np.where(np.isclose(ptp, min_val)) + p1
        dias_idx.append(min_idx.tolist())
    
    BP_features = [sys_idx, sys_val, dias_idx, dias_val]
    return BP_features


# In[4]:


def combine_data(data1, data2, data3):
    import pandas as pd
    import numpy as np
    
    ## Creating a tuple of the variable tuples following the format described above
    data = ()
    #  ECG_features = [ECG_norm, p_peak, r_peak, t_peak, q_idx, q_pt, s_idx, s_pt]
    data = data + ((data1[1], data1[0][data1[1]]),)
    data = data + ((data1[2], data1[0][data1[2]]),)
    data = data + ((data1[3], data1[0][data1[3]]),)
    data = data + ((data1[4], data1[5]),)
    data = data + ((data1[6], data1[7]),)
    # PPG_features = [ppg_valley_point_idx,ppg_valley_point,dppg_valley_point_idx,dppg_valley_point,sdppg_valley_point_idx,sdppg_valley_point,systolic_peak_idx,systolic_peak,slopeval_ppg_idx,slopeval_ppg,slopeval_dppg_idx,slopeval_dppg,dicrotic_notch_idx,dicrotic_notch,dDA_point_idx,dDA_point,sdDA_point_idx,sdDA_point,dppg_peak_idx,dppg_peak,sdppg_peak_idx,sdppg_peak]
    data = data + ((data2[0], data2[1]),)
    data = data + ((data2[2], data2[3]),)
    data = data + ((data2[4], data2[5]),)
    data = data + ((data2[6], data2[7]),)
    data = data + ((data2[8], data2[9]),)
    data = data + ((data2[10], data2[11]),)
    data = data + ((data2[12], data2[13]),)
    data = data + ((data2[14], data2[15]),)
    data = data + ((data2[16], data2[17]),)
    data = data + ((data2[18], data2[19]),)
    data = data + ((data2[20], data2[21]),)
    # ABP_features = [sys_idx, sys_val, dias_idx, dias_val]
    data = data + ((data3[0], data3[1]),)
    data = data + ((data3[2], data3[3]),)
    
    # Combining ECG wave points
    ECG_vals = []
    for i in range(len(data[0][0])):
        row_vals = []
        row_vals.extend([data[0][0][i],data[0][1][i]])
        for j in range(1,5):
            idx_index = list(x for x in data[j][0] if data[0][0][i] <= x <= data[0][0][i+1])
            if bool(idx_index) == True:
                val_index = np.where(np.asarray(data[j][0])==idx_index[0])
                val = data[j][1][val_index[0][0]]
                row_vals.extend([idx_index[0],val])
            else:
                row_vals.extend([np.nan,np.nan])
        ECG_vals = ECG_vals + [row_vals]
    
    # Combining PPG/ABP wave points
    PPG_vals = []
    for i in range(len((data[5][0]))-1):
        row_vals = []
        onset_idx = data[5][0][i]
        onset_val = data[5][1][i]
        offset_idx = data[5][0][i+1]
        offset_val = data[5][1][i+1]
        row_vals.extend([onset_idx,onset_val,offset_idx,offset_val])

        for j in range(6,18):
            idx_index = list(x for x in data[j][0] if onset_idx <= x <= offset_idx)
            if bool(idx_index) == True:
                val_index = np.where(np.asarray(data[j][0])==idx_index[0])
                val = data[j][1][val_index[0][0]]
                row_vals.extend([idx_index[0],val])
            else:
                row_vals.extend([np.nan,np.nan])
        PPG_vals = PPG_vals + [row_vals]
        
    df = pd.DataFrame(columns=[ 'p_peak_idx','p_peak',
                                'r_peak_idx','r_peak',
                                't_peak_idx','t_peak',
                                'q_idx','q_pt',
                                's_idx','s_pt',
                                'onset_valley_point_idx','onset_valley_point',
                                'offset_valley_point_idx','offset_valley_point',
                                'dppg_onset_point_idx','dppg_onset_point',
                                'sdppg_onset_point_idx','sdppg_onset_point',
                                'systolic_peak_idx','systolic_peak',
                                'slopeval_ppg_idx','slopeval_ppg',
                                'slopeval_dppg_idx','slopeval_dppg',
                                'dicrotic_notch_idx','dicrotic_notch',
                                'dDA_point_idx','dDA_point',
                                'sdDA_point_idx','sdDA_point',
                                'dppg_peak_idx', 'dppg_peak',
                                'sdppg_peak_idx', 'sdppg_peak',
                                'sys_idx','sys_val',
                                'dias_idx','dias_val'                           
                              ])
    # Matching ECG to PPG/ABP and adding to dataframe
    for i in range(len(ECG_vals)):
        for j in range(len(PPG_vals)):
            if abs(ECG_vals[i][2] - PPG_vals[j][8]) < 25:
                df.loc[i] = ECG_vals[i] + PPG_vals[j]
   
    df=df.dropna()
    return df

def combine_data1(data1, data2, range_val):
    import pandas as pd
    import numpy as np
    
    ## Creating a tuple of the variable tuples following the format described above
    data = ()
    #  ECG_features = [ECG_norm, p_peak, r_peak, t_peak, q_idx, q_pt, s_idx, s_pt]
    data = data + ((data1[1], data1[0][data1[1]]),)
    data = data + ((data1[2], data1[0][data1[2]]),)
    data = data + ((data1[3], data1[0][data1[3]]),)
    data = data + ((data1[4], data1[5]),)
    data = data + ((data1[6], data1[7]),)
    # PPG_features = [ppg_valley_point_idx,ppg_valley_point,dppg_valley_point_idx,dppg_valley_point,sdppg_valley_point_idx,sdppg_valley_point,systolic_peak_idx,systolic_peak,slopeval_ppg_idx,slopeval_ppg,slopeval_dppg_idx,slopeval_dppg,dicrotic_notch_idx,dicrotic_notch,dDA_point_idx,dDA_point,sdDA_point_idx,sdDA_point,dppg_peak_idx,dppg_peak,sdppg_peak_idx,sdppg_peak]
    data = data + ((data2[0], data2[1]),)
    data = data + ((data2[2], data2[3]),)
    data = data + ((data2[4], data2[5]),)
    data = data + ((data2[6], data2[7]),)
    data = data + ((data2[8], data2[9]),)
    data = data + ((data2[10], data2[11]),)
    data = data + ((data2[12], data2[13]),)
    data = data + ((data2[14], data2[15]),)
    data = data + ((data2[16], data2[17]),)
    data = data + ((data2[18], data2[19]),)
    data = data + ((data2[20], data2[21]),)
    
    
    # Combining ECG wave points
    ECG_vals = []
    for i in range(len(data[0][0])-1):
        row_vals = []
        row_vals.extend([data[0][0][i],data[0][1][i]])
        for j in range(1,5):
            idx_index = list(x for x in data[j][0] if data[0][0][i] <= x <= data[0][0][i+1])
            if bool(idx_index) == True:
                val_index = np.where(np.asarray(data[j][0])==idx_index[0])
                val = data[j][1][val_index[0][0]]
                row_vals.extend([idx_index[0],val])
            else:
                row_vals.extend([np.nan,np.nan])
        ECG_vals = ECG_vals + [row_vals]
    
    # Combining PPG wave points
    PPG_vals = []
    for i in range(len((data[5][0]))-1):
        row_vals = []
        onset_idx = data[5][0][i]
        onset_val = data[5][1][i]
        offset_idx = data[5][0][i+1]
        offset_val = data[5][1][i+1]
        row_vals.extend([onset_idx,onset_val,offset_idx,offset_val])

        for j in range(6,16):
            idx_index = list(x for x in data[j][0] if onset_idx <= x <= offset_idx)
            if bool(idx_index) == True:
                val_index = np.where(np.asarray(data[j][0])==idx_index[0])
                val = data[j][1][val_index[0][0]]
                row_vals.extend([idx_index[0],val])
            else:
                row_vals.extend([np.nan,np.nan])
        PPG_vals = PPG_vals + [row_vals]
        
    df = pd.DataFrame(columns=[ 'p_peak_idx','p_peak',
                                'r_peak_idx','r_peak',
                                't_peak_idx','t_peak',
                                'q_idx','q_pt',
                                's_idx','s_pt',
                                'onset_valley_point_idx','onset_valley_point',
                                'offset_valley_point_idx','offset_valley_point',
                                'dppg_onset_point_idx','dppg_onset_point',
                                'sdppg_onset_point_idx','sdppg_onset_point',
                                'systolic_peak_idx','systolic_peak',
                                'slopeval_ppg_idx','slopeval_ppg',
                                'slopeval_dppg_idx','slopeval_dppg',
                                'dicrotic_notch_idx','dicrotic_notch',
                                'dDA_point_idx','dDA_point',
                                'sdDA_point_idx','sdDA_point',
                                'dppg_peak_idx', 'dppg_peak',
                                'sdppg_peak_idx', 'sdppg_peak',                          
                              ])
    # Matching ECG to PPG and adding to dataframe
    for i in range(len(ECG_vals)):
        for j in range(len(PPG_vals)):
            if abs(ECG_vals[i][2] - PPG_vals[j][8]) < range_val:
                df.loc[i] = ECG_vals[i] + PPG_vals[j]
   
    df=df.dropna()
    return df

# In[5]:


# FEATURE EQUATION DEFINITIONS
def delta_time(idx_a, idx_b):
    return abs((idx_b-idx_a)/125)

def slope(val_a, val_b, idx_a, idx_b): 
    return abs((val_b-val_a)/delta_time(idx_a,idx_b)) 

def auc(data, idx_a, idx_b):
    import numpy as np
    x = data[int(idx_a):int(idx_b+1)]
    y = range(0, int(idx_b-idx_a)+1)
    return abs(np.trapz(y=y, x=x))


# In[6]:
import numpy as np

def features(df,PPG_norm):
    # QRS_time: Time for a full QRS wave (ECG)
    df['QRS_time'] = df.apply(lambda row : delta_time(row['q_idx'],row['s_idx']), axis = 1) 
    
    # RP_time: Time from P peak to R peak (ECG)
    df['RP_time'] = df.apply(lambda row : delta_time(row['p_peak_idx'],row['r_peak_idx']), axis = 1)
    
    # RT_time: Time from R peak to T peak (ECG)
    df['RT_time'] = df.apply(lambda row : delta_time(row['r_peak_idx'],row['t_peak_idx']), axis = 1)  

    # PQ_time: Time from P peak to Q valley (ECG)
    df['PQ_time'] = df.apply(lambda row : delta_time(row['p_peak_idx'],row['q_idx']), axis = 1)  

    # ST_time: Time from S valley to T peak (ECG)
    df['ST_time'] = df.apply(lambda row : delta_time(row['s_idx'],row['t_peak_idx']), axis = 1)  

    # PT_time: Time from P peak to T peak (ECG)
    df['PT_time'] = df.apply(lambda row : delta_time(row['p_peak_idx'],row['t_peak_idx']), axis = 1)  

    # P_amp: P peak amplitude
    df['P_amp'] = df.apply(lambda row : row['p_peak'], axis = 1) 

    # R_amp: R peak amplitude
    df['R_amp'] = df.apply(lambda row : row['r_peak'], axis = 1) 

    # T_amp: T peak amplitude
    df['T_amp'] = df.apply(lambda row : row['t_peak'], axis = 1) 
    
    # RT_ratio: T_amp/R_amp
    df['RT_ratio'] = df.apply(lambda row : row['t_peak']/row['r_peak'] if row['r_peak'] != 0 else np.na, axis = 1) 

    # RP_diff: R_amp - P_amp
    df['RP_diff'] = df.apply(lambda row : row['r_peak']-row['p_peak'], axis = 1) 

    # PTT_p: Time from R peak of ECG to systolic peak of PPG
    df['PTT_p'] = df.apply(lambda row : delta_time(row['r_peak_idx'],row['systolic_peak_idx']), axis = 1)

    # PTT_d: Time from R peak of ECG to max slope point of PPG (dPPG)
    df['PTT_d'] = df.apply(lambda row : delta_time(row['r_peak_idx'],row['dppg_peak_idx']), axis = 1)

    # PTT_f: Time from R peak of ECG to foot of PPG signal
    df['PTT_f'] = df.apply(lambda row : delta_time(row['r_peak_idx'],row['offset_valley_point_idx']), axis = 1)

    # HR: Heart Rate (Peak-to-Peak time --> BPM)
    # (idx_2 - idx_1) - no. of samples in a beat
    # 125 samples/sec * 60 sec/min => samples/min / # of samples = BPM
    df['HR'] = None
    for i in range(len(df)-1):
        if delta_time(df['systolic_peak_idx'].iloc[i], df['systolic_peak_idx'].iloc[i+1]) != 0:
            df['HR'].iloc[i] = 1*60/delta_time(df['systolic_peak_idx'].iloc[i], df['systolic_peak_idx'].iloc[i+1])
    
    # AS: Ascending slope of PPG (slope from onset point to max peak)
    # (systolic peak - onset point)/(delta(time))
    df['AS'] = df.apply(lambda row : slope(row['onset_valley_point'],row['systolic_peak'],row['onset_valley_point_idx'],row['systolic_peak_idx']) if delta_time(row['onset_valley_point_idx'],row['systolic_peak_idx']) != 0 else np.nan, axis = 1)  

    # DS: Descending slope of PPG (slope from max peak to offset point)
    # (offset point - systolic peak)/(delta(time))
    df['DS'] = df.apply(lambda row : slope(row['systolic_peak'],row['offset_valley_point'],row['systolic_peak_idx'],row['offset_valley_point_idx']) if delta_time(row['systolic_peak_idx'],row['offset_valley_point_idx']) != 0 else np.nan, axis = 1)  

    # S1: Area under curve between onset and max slope point
    df['S1'] = df.apply(lambda row : auc(PPG_norm, row['onset_valley_point_idx'], row['slopeval_ppg_idx']), axis=1)

    # S2: Area under curve between max slope point and max peak
    df['S2'] = df.apply(lambda row : auc(PPG_norm, row['slopeval_ppg_idx'], row['systolic_peak_idx']), axis=1)

    # S3: Area under curve between max peak and dicrotic notch
    df['S3'] = df.apply(lambda row : auc(PPG_norm, row['systolic_peak_idx'], row['dicrotic_notch_idx']), axis=1)

    # S4: Area under curve between dicrotic notch and offset point
    df['S4'] = df.apply(lambda row : auc(PPG_norm, row['dicrotic_notch_idx'], row['offset_valley_point_idx']), axis=1)

    # AA: Ascending area of PPG
    df['AA'] = df.apply(lambda row: (row['S1'] + row['S2']), axis=1)
    
    # DA: Descending area of PPG
    df['DA'] = df.apply(lambda row: (row['S3'] + row['S4']), axis=1)

    # PI: Peak intensity of PPG
    df['PI'] = df.apply(lambda row : row['systolic_peak'], axis=1)

    # dPI: Peak intensity of dPPG
    df['dPI'] = df.apply(lambda row : row['dppg_peak'], axis=1)

    # sdPI: Peak intensity of sdPPG
    df['sdPI'] = df.apply(lambda row : row['sdppg_peak'], axis=1)

    # dVI: Valley intensity of dPPG
    df['dVI'] = df.apply(lambda row : row['dppg_onset_point'], axis=1)

    # sdVI: Valley intensity of sdPPG
    df['sdVI'] = df.apply(lambda row : row['sdppg_onset_point'], axis=1)

    # AID: Intensity diff between max peak and onset point (PPG)
    df['AID'] = df.apply(lambda row: (row['systolic_peak'] - row['onset_valley_point']), axis=1)

    # dAID: Intensity diff between max peak and onset point (dPPG)
    df['dAID'] = df.apply(lambda row: (row['dppg_peak'] - row['dppg_onset_point']), axis=1)

    # sdAID: Intensity diff between max peak and onset point (sdPPG)
    df['sdAID'] = df.apply(lambda row: (row['sdppg_peak'] - row['sdppg_onset_point']), axis=1)

    # dDID: Intensity diff between offset point and max peak (dPPG)
    df['dDID'] = df.apply(lambda row: (row['dppg_peak'] - row['offset_valley_point']), axis=1)

    # sdDID: Intensity diff between offset point and max peak (sdPPG)
    df['sdDID'] = df.apply(lambda row: (row['sdppg_peak'] - row['offset_valley_point']), axis=1)

    # dRIPV: Ratio of max peak to valley intensity (dPPG)
    df['dRIPV'] = df.apply(lambda row: (row['dppg_peak'] / row['dppg_onset_point']) if row['dppg_onset_point'] != 0 else np.nan, axis=1)

    # sdRIPV: Ratio of max peak to valley intensity (sdPPG)
    df['sdRIPV'] = df.apply(lambda row: (row['sdppg_peak'] / row['sdppg_onset_point']) if row['sdppg_onset_point'] != 0 else np.nan, axis=1)

    # AT: Ascending time interval of PPG
    df['AT'] = df.apply(lambda row: delta_time(row['onset_valley_point_idx'], row['systolic_peak_idx']), axis=1)

    # Slope_a: Slope from max peak to dicrotic notch of PPG
    df['slope_a'] = df.apply(lambda row: slope(row['systolic_peak'], row['dicrotic_notch'], row['systolic_peak_idx'], row['dicrotic_notch_idx']) if delta_time(row['systolic_peak_idx'],row['dicrotic_notch_idx']) != 0 else np.nan, axis=1)
    # NI: Dicrotic notch intensity
    df['NI'] = df.apply(lambda row : row['dicrotic_notch'], axis=1)

    # AI: Augmentation index = NI/PI
    df['AI'] = df.apply(lambda row: (row['NI'] / row['PI']) if row['PI'] != 0 else np.nan, axis=1)

    # AI1: Augmentation index 1 = (PI-NI)/PI
    df['AI1'] = df.apply(lambda row: abs((row['PI'] - row['NI']) / row['PI']) if row['PI'] != 0 else np.nan, axis=1)

    # RSD: Ratio of systolic to diastolic duration
    df['RSD'] = df.apply(lambda row: (delta_time(row['onset_valley_point_idx'], row['dicrotic_notch_idx']) / delta_time(row['dicrotic_notch_idx'], row['offset_valley_point_idx'])) if delta_time(row['dicrotic_notch_idx'], row['offset_valley_point_idx']) != 0 else np.nan, axis=1)

    # RSC: Ratio of diastolic duration to cardiac cycle
    df['RSC'] = df.apply(lambda row: (delta_time(row['dicrotic_notch_idx'], row['offset_valley_point_idx']) / delta_time(row['onset_valley_point_idx'], row['offset_valley_point_idx'])) if delta_time(row['onset_valley_point_idx'], row['offset_valley_point_idx']) != 0 else np.nan, axis=1)

    # RDC: Ratio of systolic duaration to cardiac cycle
    df['RDC'] = df.apply(lambda row: (delta_time(row['onset_valley_point_idx'], row['dicrotic_notch_idx']) / delta_time(row['onset_valley_point_idx'], row['offset_valley_point_idx'])) if delta_time(row['onset_valley_point_idx'], row['offset_valley_point_idx']) != 0 else np.nan, axis=1)
    
    df=df.dropna()
    return df

