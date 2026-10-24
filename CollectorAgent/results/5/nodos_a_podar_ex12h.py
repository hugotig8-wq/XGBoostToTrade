import json

with open('clf_xgboost_return_12h3.json', 'r') as f:
    data = json.load(f)

trees = data['learner']['gradient_booster']['model']['trees']
print(f"Total de árboles: {len(trees)}")

nodos_hijos_inutiles = 0
nodos_baja_confianza = 0

for tree in trees:
    # Contar cuántas ramas terminan en hoja con pérdida 0 (loss_changes == 0.0)
    nodos_hijos_inutiles += sum(1 for loss in tree['loss_changes'] if loss == 0.0)
    
    # Contar hojas con muy pocos datos (sum_hessian < 1.5)
    # Las hojas se identifican porque su left_children es -1
    for i, child in enumerate(tree['left_children']):
        if child == -1 and tree['sum_hessian'][i] < 1.5:
            nodos_baja_confianza += 1

print(f"Ramas inútiles (loss=0): {nodos_hijos_inutiles}")
print(f"Hojas sobreajustadas (hessian < 1.5): {nodos_baja_confianza}")
