import streamlit as st
import pandas as pd 
import lasio 
from io import StringIO
import altair as alt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.title('Gradiente de Temperatura')
st.sidebar.title('Paramétros')
@st.cache
def lectura(archivo_las):
	try:
		bytes_data = archivo_las.read()
		str_io = StringIO(bytes_data.decode('Windows-1252'))
		las_file = lasio.read(str_io)
		df = las_file.df()
		df['DEPTH'] = df.index
	except:
		st.sidebar.error("Archivo no admitido ") # Rojo
	return df, las_file
@st.cache(allow_output_mutation=True)
def dataframe_limitado(filtered_df, limite_superior,limite_inferior):
	try:
		df_limitado=df[limite_superior:limite_inferior]
	except:
		st.write('DataFrame no pudo ser limitado')
	return df_limitado
archivo_las = st.sidebar.file_uploader("Cargar archivo LAS" , type=['.las', '.LAS'], key=None)
if archivo_las is None:
	st.sidebar.warning("Suba un archivo con extencion .las") #Amarillo
else:
	progreso = st.sidebar.progress(0)
	df, las_file = lectura(archivo_las)
	progreso.progress(100)
	st.sidebar.success("Archivo las cargado exitosamente") # Verde
	st.write(df)
	with st.beta_expander ('Informacion'):
		nombre_pozo = las_file.header['Well'].WELL.value
		#pais = las_file.header['Well'].COUNT.value
		campo = las_file.header['Well'].FLD.value
		#provincia = las_file.header['Well'].PROV.value
		compania = las_file.header['Well'].COMP.value
		unidades_profundidad = las_file.header['Well'].STRT.unit
		profundidad_min = df.index.values[0]
		profundidad_max = df.index.values[-1]
		cola , colb = st.beta_columns(2)
		with cola:
			st.write('Prufundidad inicial: {} {}'.format(profundidad_min,unidades_profundidad))
			st.write('Prufundidad final: {} {}'.format(profundidad_max, unidades_profundidad))
			st.write('Nombre del pozo: {}'.format(nombre_pozo))
			st.write('Nombre del campo: {}'.format(campo))
		with colb:
			#st.write('Pais: {}'.format(pais))
			#st.write('Provincia: {}'.format(provincia))
			st.write('Compania: {}'.format(compania))
			st.write('Unidad profundiad: {}'.format(unidades_profundidad))
	with st.beta_expander ('Analisis del registro'):
		lista_registros =list(df.columns)
		col1, col2, col3 = st.beta_columns(3)
		with col1:
			df_columnas=pd.DataFrame(lista_registros, columns = ['Lista de registros'])
			st.write(df_columnas)
		with col2:
			registro = st.selectbox('seleccione el resgitro de temperatura', options= lista_registros)
			df= df[['DEPTH',registro]]
			df_mask=df[registro] >= 0
			filtered_df = df[df_mask]
		with col3:
			fig , ax = plt.subplots(figsize=(3,5))
			ax.plot(filtered_df[registro], filtered_df.DEPTH)
			ax.invert_yaxis()
			ax.set_ylabel("Depth")
			ax.set_xlabel(registro)
			ax.grid()
			st.pyplot(fig)
		grafico = alt.Chart(filtered_df).mark_line().encode(
		alt.X('DEPTH:Q',scale=alt.Scale(domain=(profundidad_min, profundidad_max))),
		alt.Y(registro, axis = alt.Axis(labelOverlap = True)),
		tooltip=[registro,'DEPTH']).interactive().properties(width=800, height = 100)
		st.altair_chart(grafico)
	with st.beta_expander ('Regresión lineal'):
		col4, col5, col6 = st.beta_columns(3)
		with  col4:
			st.write(filtered_df)
		with col5 :
			df_filtrado_profundidad_min = filtered_df.index.values[0]
			df_filtrado_profundidad_max = filtered_df.index.values[-1]
			limite_superior = st.number_input('Igrese limite superior',min_value=df_filtrado_profundidad_min, max_value=None, value=df_filtrado_profundidad_min,  step=1.00 )
			limite_inferior = st.number_input('Igrese limite inferior',min_value=0.00, max_value=df_filtrado_profundidad_max, value=df_filtrado_profundidad_max,  step=1.00 )
			df_limitado = dataframe_limitado(filtered_df, limite_superior , limite_inferior)
		with col6 :
			st.write(df_limitado)
	with st.beta_expander ('Resultados'):
		df_pred = df_limitado

		y = df_pred[[registro]]
		x = df_pred[['DEPTH']]
		df_model =LinearRegression()
		df_model.fit(x,y)
		df_R_sq = df_model.score(x,y)
		df_intercepto = df_model.intercept_
		df_coeficiente = df_model.coef_

		st.write("Coeficiente de derterminacion: {}".format(round(df_R_sq , 2)))
		st.write("Intercepto (bo): {}".format(df_intercepto))
		st.write("Pendiente (b1) : {}".format(df_coeficiente))
		y_pred=df_model.predict(x)
		df_pred["Predicción"]=y_pred
		st.write(df_pred)
		graf1 = alt.Chart(df_limitado).mark_point().encode(
			alt.X('DEPTH:Q', scale=alt.Scale(domain=(limite_superior, limite_inferior))),
			alt.Y(registro),
			tooltip=[registro,'DEPTH']).interactive().properties(width=800, height = 100)
		grafico_pred = alt.Chart(df_limitado).mark_line(color='red').encode(
		    alt.X('DEPTH:Q', scale=alt.Scale(domain=(limite_superior, limite_inferior))),
		    alt.Y('Predicción'),
		).interactive().interactive().properties(width=800, height = 100)
		graficos = alt.layer(graf1,grafico_pred)

		st.altair_chart(graf1)
		st.altair_chart(grafico_pred)
		st.altair_chart(graficos)

	with st.beta_expander ('Cálculo'):
		st.info("Ecuación: (Tf-To/Df-Do)") # Azul
		col7, col8 = st.beta_columns(2)
		with col7:
			st.write(df_pred)

		with col8:
			try:
				st.warning("Datos:") #Amarillo

				To = df_pred.iloc[0]['Predicción']
				Tf = df_pred.iloc[-1]['Predicción']
				Do = df_pred.iloc[0]['DEPTH']
				Df = df_pred.iloc[-1]['DEPTH']

				st.write('To: {}'.format(To))
				st.write('Tf: {}'.format(Tf))


				st.write('Do: {}'.format(Do))
				st.write('Df: {}'.format(Df))


				dt = (Tf-To)/(Df-Do)
				st.write('Gradiente',dt, '°/{}'.format(unidades_profundidad))

			except:
				st.error('Debe seleccionar el registro de temperatura para determinar el gradiente geotérmico')



#st.sidebar.info("Suba un archivo con extencion .las") # Azul
#st.sidebar.error("Suba un archivo con extencion .las") # Rojo
#st.sidebar.warning("Suba un archivo con extencion .las") #Amarillo
