import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/Fa11ingDeep/Data-Set/main/youtube_spotify_limpio.csv"
df_songs = pd.read_csv(url, encoding="UTF-8")

# Convertir el diccionario en dataframe
torch.cuda.is_available()
# Eliminar los valores nulos
df_songs = df_songs.dropna()
# Fijar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
# Fijar semilla para reproducibilidad
torch.manual_seed(2)
torch.cuda.manual_seed(2)
# Preparar los datos
X_np = df_songs[['Views', 'Comments', 'Stream']].values
y_np = df_songs['Likes'].values
# Escalado
X_scaled = StandardScaler().fit_transform(X_np)
y_scaled = StandardScaler().fit_transform(y_np.reshape(-1, 1))
# Convertir a tensores y pasar a GPU si esta disponible
X = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y = torch.tensor(y_scaled, dtype=torch.float32).to(device)
# Split 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)
# Definicion del modelo (ya no es linear regression, pero dejamos el mismo nombre para que quede claro que fue una mejora de ese modelo)
class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    # Se construye la red neuronal con 3 capas ocultas
    self.model = nn.Sequential(
      nn.Linear(3, 128), nn.ReLU(), nn.Dropout(0.3),
      nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
      nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
      nn.Linear(32, 1)
    )
  def forward(self, x):
    return self.model(x)
# Crear modelo
model = LinearRegressionModel().to(device)
# Funcion de perdida
loss_fn = nn.L1Loss().to(device)
# Optimizador
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Datos
history = {"epoch": [],"train_loss": [],"val_loss": [],"test_loss": [],"r2_score": [],"val_r2_score": []}
# Datos y condiciones para el early stopping
best_val_loss = float('inf')
patience = 155 # Ciclos de tolerancia
trigger_times = 0
best_model_state = None
best_epoch = 0

best_val_r2 = float('-inf')
best_r2_model_state = None
best_r2_epoch = 0
# Entrenamiento
epochs = 10000
for epoch in range(epochs):
  model.train()
  y_pred = model(X_train)
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  # Evaluacion
  model.eval()
  with torch.inference_mode():
    val_pred = model(X_val)
    val_loss = loss_fn(val_pred, y_val)
    val_r2 = r2_score(y_val.cpu().numpy(), val_pred.cpu().numpy())
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)
    test_r2 = r2_score(y_test.cpu().numpy(), test_pred.cpu().numpy())
  # Early stopping
  if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    trigger_times = 0
    best_model_state = model.state_dict()  # Guardar el mejor modelo
    best_epoch = epoch
  else:
    trigger_times += 1
    if trigger_times >= patience:
      break
  # Guardamos el modelo por si no ocurre el early stopping
  if val_r2 > best_val_r2:
    best_val_r2 = val_r2
    best_r2_model_state = model.state_dict()
    best_r2_epoch = epoch
  # Guardar cada 10 ciclos
  if epoch % 10 == 0:
    history["epoch"].append(epoch)
    history["train_loss"].append(loss.item())
    history["val_loss"].append(val_loss.item())
    history["test_loss"].append(test_loss.item())
    history["r2_score"].append(test_r2)
    history["val_r2_score"].append(val_r2)
# Restaurar el mejor modelo y realizar la evaluacion
if best_model_state:
  model.load_state_dict(best_model_state)
else:
  model.load_state_dict(best_r2_model_state)
model.eval()
with torch.inference_mode():
  final_val_pred = model(X_val)
  final_test_pred = model(X_test)
final_val_r2 = r2_score(y_val.cpu().numpy(), final_val_pred.cpu().numpy())
final_test_r2 = r2_score(y_test.cpu().numpy(), final_test_pred.cpu().numpy())
final_val_loss = loss_fn(final_val_pred, y_val).item()
final_test_loss = loss_fn(final_test_pred, y_test).item()
last_epoch = max(history["epoch"])
print(last_epoch)


history_df = pd.DataFrame(history)
plt.figure(figsize=(7, 5))
# Graficar las lineas
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["test_loss"], label="Test Loss")
plt.plot(history_df["epoch"], history_df["r2_score"], label="R2 Score")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
plt.plot(history_df["epoch"], history_df["val_r2_score"], label="Val R2 Score")
# Buscar R2 maximo, train loss, validation loss minimo y test loss minimo y los marca como un punto en el grafico
# Se agrega los valores especificos en el label
epoch_to_plot = best_epoch if best_model_state else max(history["epoch"])

plt.scatter(epoch_to_plot, final_val_loss, color="red", marker="o", s=50, label=f"Val Loss (Epoch {epoch_to_plot}) = {final_val_loss:.4f}")
plt.scatter(epoch_to_plot, final_val_r2, color="purple", marker="o", s=50, label=f"Val R² (Epoch {epoch_to_plot}) = {final_val_r2:.4f}")
plt.scatter(epoch_to_plot, final_test_loss, color="orange", marker="o", s=50, label=f"Test Loss (Epoch {epoch_to_plot}) = {final_test_loss:.4f}")
plt.scatter(epoch_to_plot, final_test_r2, color="green", marker="o", s=50, label=f"Test R² (Epoch {epoch_to_plot}) = {final_test_r2:.4f}")

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()