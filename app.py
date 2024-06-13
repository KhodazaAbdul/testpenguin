import joblib
import pandas as pd
import streamlit as st
st.title('app di ML Penguins')
model_pipe = joblib.load('penguinspipe.pkl')
print('modello caricato')

island= st.selectbox('inserire isola', ['Torgensen', 'Dream','Biscoe'])
bill_length_mm= st.number_input('inserire lunghezza becco', 20.0, 60.0, 50.0)
bill_depth_mm=st.number_input('inserire profondità becco', 5.0,30.0,20.0)
flipper_length_mm= st.number_input('inserire lunghezza pinna',10.0, 300.0, 180.0 )
body_mass_g = st.number_input('inserire massa corporea', 2000, 7000,3000 )
sex = st.selectbox('inserire sesso',['male', 'female'])

# island = 'Torgersen'
# bill_length_mm = 20.33
# bill_depth_mm = 10.50
# flipper_length_mm = 40.25
# body_mass_g = 359.50
# sex = 'female'

data = {
        "island": [island],
        "bill_length_mm": [bill_length_mm],
        "bill_depth_mm": [bill_depth_mm],
        "flipper_length_mm": [flipper_length_mm],
        "body_mass_g": [body_mass_g],
        "sex": [sex]
        }

input_df = pd.DataFrame(data)
res = model_pipe.predict(input_df).astype(int)[0]
print(res)

classes = {0:'Adelie',
           1:'Gentoo',
           2:'Chinstrap'
           }

y_pred = classes[res]

if st.button('Predicts'):
    st.success(f'la specie predetta è {y_pred}')

#['Adelie','Gentoo', 'Chinstrap']