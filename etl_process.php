<?php
// etl_process.php - Procesa los datos y devuelve JSON
header('Content-Type: application/json');

$inputFile = 'ventas_raw.csv';
$outputFile = 'reporte_mensual.csv';

try {
    // Verificar que existe el archivo
    if (!file_exists($inputFile)) {
        throw new Exception("Archivo no encontrado: $inputFile");
    }

    // EXTRACT - Leer CSV
    $handle = fopen($inputFile, "r");
    if (!$handle) {
        throw new Exception("No se pudo abrir el archivo");
    }

    $headers = fgetcsv($handle);
    if (!$headers) {
        throw new Exception("El archivo está vacío");
    }

    $data = [];
    while (($row = fgetcsv($handle)) !== false) {
        if (count($row) === count($headers)) {
            $data[] = array_combine($headers, $row);
        }
    }
    fclose($handle);

    $originalCount = count($data);

    // TRANSFORM - Limpieza
    $cleanData = [];
    
    foreach ($data as $row) {
        // Limpiar espacios
        foreach ($row as $key => $value) {
            $row[$key] = trim($value);
        }

        // Limpiar precio
        if (isset($row['precio'])) {
            $precio = str_replace(['$', ',', ' '], '', $row['precio']);
            $row['precio'] = floatval($precio);
        }

        // Limpiar cantidad
        if (isset($row['cantidad'])) {
            $cantidad = strtolower(trim($row['cantidad']));
            $textoNumeros = [
                'uno' => 1, 'dos' => 2, 'tres' => 3, 'cuatro' => 4,
                'cinco' => 5, 'seis' => 6, 'siete' => 7, 'ocho' => 8,
                'nueve' => 9, 'diez' => 10
            ];
            
            if (isset($textoNumeros[$cantidad])) {
                $row['cantidad'] = $textoNumeros[$cantidad];
            } elseif ($cantidad === '' || $cantidad === 'null') {
                $row['cantidad'] = 1;
            } else {
                $row['cantidad'] = intval($row['cantidad']);
            }
            
            if ($row['cantidad'] <= 0) {
                $row['cantidad'] = 1;
            }
        }

        // Normalizar texto
        $textFields = ['producto', 'cliente', 'region'];
        foreach ($textFields as $field) {
            if (isset($row[$field]) && $row[$field] !== '') {
                $row[$field] = ucwords(strtolower($row[$field]));
            }
        }

        // Formatear fecha
        if (isset($row['fecha']) && $row['fecha'] !== '') {
            $fecha = date_create($row['fecha']);
            if ($fecha) {
                $row['fecha'] = date_format($fecha, 'Y-m-d');
            } else {
                continue; // Ignorar fila con fecha inválida
            }
        } else {
            continue; // Ignorar fila sin fecha
        }

        // Validar campos obligatorios
        if (!empty($row['producto']) && !empty($row['cliente'])) {
            $cleanData[] = $row;
        }
    }

    // Eliminar duplicados
    $uniqueData = [];
    $seenKeys = [];
    $duplicatesCount = 0;

    foreach ($cleanData as $row) {
        $key = mb_strtolower($row['producto']) . '|' . 
               mb_strtolower($row['cliente']) . '|' . 
               $row['fecha'];
        
        if (!isset($seenKeys[$key])) {
            $seenKeys[$key] = true;
            $uniqueData[] = $row;
        } else {
            $duplicatesCount++;
        }
    }

    $finalCount = count($uniqueData);

    // LOAD - Guardar CSV limpio
    $out = fopen($outputFile, "w");
    if (!$out) {
        throw new Exception("No se pudo crear el archivo de salida");
    }

    fputcsv($out, $headers);
    foreach ($uniqueData as $row) {
        fputcsv($out, $row);
    }
    fclose($out);

    // Calcular reducción
    $reduction = $originalCount > 0 ? round((($originalCount - $finalCount) / $originalCount) * 100, 2) : 0;

    // Respuesta JSON exitosa
    echo json_encode([
        'success' => true,
        'original' => $originalCount,
        'duplicates' => $duplicatesCount,
        'final' => $finalCount,
        'reduction' => $reduction,
        'file' => $outputFile
    ]);

} catch (Exception $e) {
    // Respuesta JSON de error
    echo json_encode([
        'success' => false,
        'message' => $e->getMessage()
    ]);
}
?>