import numpy as np
import pandas as pd

# Class for input and output preprocessing
class Preprocessor:
    
    # returns the numpy array representing Speech and Non speech value where
    # chunk size => millis (default value of millis = 20 ms)
    def preprocess_output_file(self, filename, millis = 20):
        data = pd.read_csv(filename)
        #print(data)
        
        # Converting seconds into millis
        data["tmi0"] = data["tmi0"] * 1000
        data["tmax"] = data["tmax"] * 1000
        #print(data)
        
        start_val = 0.0
        result = np.empty((0,1))
        
        for index, row in data.iterrows():
            
            # row["text"]: 1 if speech and 0 if non-speech
            state = row["text"]
            
            # complete chunks
            diff = row["tmax"] - start_val
            no_of_complete_chunks = int(diff/millis)
            #print(no_of_complete_chunks)
            if(state == 0):
                arr_to_append = np.zeros((no_of_complete_chunks,1))
            elif(state == 1):
                arr_to_append = np.ones((no_of_complete_chunks,1))
                
            result = np.append(result, arr_to_append, axis=0)
                
            # overlapping chunks
            extra = diff - (no_of_complete_chunks * millis)
            if(extra > 0):
                hf = millis/2   # if millis = 20ms the hf is 10
                # if millis = 20ms, extra >= 10ms and currently we are on SPEECH(state=1)
                # then we'll consider appending 1 otherwise 0
                # So, the below condition takes care of assignining proper state value(whether
                # it is speech or non_speech based on the overlapping duration)
                # Speech OR Nonspeech value is assigned considering the major chunk of
                # overlapping data
                if((extra >= hf and state == 0) or (extra < hf and state == 1)):
                    arr_to_append = np.zeros((1,1))
                elif((extra >= hf and state == 1) or (extra < hf and state == 0)):
                    arr_to_append = np.ones((1,1))
                result = np.append(result, arr_to_append, axis=0)
                start_val = start_val + ((no_of_complete_chunks * millis) + millis)
            else:
                start_val = start_val + (no_of_complete_chunks * millis)
            
            #print(start_val)
        
        #print(result.shape)
        return result
    
    
# Test (How to use?)
pp = Preprocessor()
filename = "canvas_initial_files\KO_CSV_FINAL.csv"
result = pp.preprocess_output_file(filename)
print(result)