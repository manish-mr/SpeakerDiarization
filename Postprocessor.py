import pandas as pd

# Class for input and output preprocessing
class Postprocessor:
    
    # Convert y_predicted output to csv file
    # 'tier' represents the channel name
    def process_output(self, data, filename, tier, millis=20, threshold=5):
        result = pd.DataFrame(columns=['tmi0','tmax','text','tier'])
        
        current_state = data[0][0]
        start_time = 0.0
        cnt = 0.0
        threshold_cnt = 0.0
        j = 0
        
        for i in range(data.shape[0]):
            if(current_state == data[i][0]):
                cnt += 1
                if(threshold_cnt != 0):
                    threshold_cnt = 0
            elif(threshold_cnt <= threshold):
                threshold_cnt += 1
                cnt += 1
            else:
                threshold_cnt += 1
                cnt += 1
                time_in_millis = (cnt - threshold_cnt) * millis
                end_time = start_time + time_in_millis
                result.loc[j] = [start_time/1000, end_time/1000, int(current_state), tier]
                j += 1
                
                start_time = end_time
                cnt = threshold_cnt
                threshold_cnt = 0
                current_state = data[i][0]
        
        # last record
        time_in_millis = cnt * millis
        end_time = start_time + time_in_millis
        result.loc[j] = [start_time/1000, end_time/1000, int(current_state), tier]
        
        result.to_csv(filename, index=False)