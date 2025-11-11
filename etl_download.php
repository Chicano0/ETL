<?php
// ========================================
// ETL DOWNLOAD - Descarga archivo limpio
// ========================================

$outputFile = 'reporte_mensual.csv';

// Verificar que el archivo existe
if (!file_exists($outputFile)) {
    http_response_code(404);
    die("❌ Error: El archivo no existe. Por favor procesa los datos primero.");
}

// Verificar que el archivo no está vacío
if (filesize($outputFile) === 0) {
    http_response_code(500);
    die("❌ Error: El archivo está vacío. Por favor procesa los datos nuevamente.");
}

// Limpiar cualquier salida previa
if (ob_get_level()) {
    ob_end_clean();
}

// Nombre del archivo con fecha y hora
$downloadName = 'reporte_mensual_' . date('Y-m-d_His') . '.csv';

// Headers para forzar descarga
header('Content-Type: text/csv; charset=utf-8');
header('Content-Disposition: attachment; filename="' . $downloadName . '"');
header('Content-Length: ' . filesize($outputFile));
header('Cache-Control: no-cache, must-revalidate');
header('Pragma: no-cache');
header('Expires: 0');

// Enviar el archivo al navegador
readfile($outputFile);

// Terminar ejecución
exit;
?>