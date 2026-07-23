from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def entrenar_modelo(X, y):
    # Entrenamos un modelo de Random Forest usando la librería sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)

    # Métrica de interés. Preveo añadir más métricas en el futuro, pero por ahora nos quedamos con MAE.
    print(f"Error Absoluto Medio (MAE): {mae:.2f} vehículos")
    
    return modelo