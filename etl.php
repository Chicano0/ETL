<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETL - Limpieza de Datos</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding: 40px 20px;
        }
        
        .card {
            max-width: 600px;
            margin: 0 auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <center>
    <div class="container">
        <br>
        <br>
        <br>
        <br>
        <br>
        <br>
        <br>
        <br>
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Limpieza de Datos ETL</h5>
                <p class="card-text">Procesa y limpia tu archivo <strong>ventas_raw.csv</strong> eliminando duplicados y normalizando los datos.</p>
                
                <button class="btn btn-primary" id="processBtn" onclick="processETL()">
                    Limpiar Datos
                </button>
                
                <div class="mt-3" id="loading" style="display: none;">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Cargando...</span>
                    </div>
                    <span class="ms-2">Procesando...</span>
                </div>
            </div>
        </div>
    </div>
    </center>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>

    <script>
        function processETL() {
            const btn = document.getElementById('processBtn');
            const loading = document.getElementById('loading');
            
            btn.disabled = true;
            loading.style.display = 'block';
            
            fetch('etl_process.php')
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    btn.disabled = false;
                    
                    if (data.success) {
                        Swal.fire({
                            icon: 'success',
                            title: 'Datos Limpiados',
                            html: `
                                <div style="text-align: left;">
                                    <p><strong>Filas originales:</strong> ${data.original}</p>
                                    <p><strong>Duplicados eliminados:</strong> ${data.duplicates}</p>
                                    <p><strong>Filas finales:</strong> ${data.final}</p>
                                    ${data.reduction > 0 ? `<p><strong>Reducción:</strong> ${data.reduction}%</p>` : ''}
                                </div>
                            `,
                            confirmButtonText: 'Descargar Archivo',
                            showCloseButton: true
                        }).then((result) => {
                            if (result.isConfirmed) {
                                window.location.href = 'etl_download.php';
                            }
                        });
                    } else {
                        Swal.fire({
                            icon: 'error',
                            title: 'Error',
                            text: data.message || 'No se pudo procesar el archivo.'
                        });
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    btn.disabled = false;
                    
                    Swal.fire({
                        icon: 'error',
                        title: 'Error de Conexión',
                        text: 'No se pudo conectar con el servidor.'
                    });
                });
        }
    </script>
</body>
</html>