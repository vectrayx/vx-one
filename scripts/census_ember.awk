# Censo EMBER2024 Win32 — label + familia, por archivo (paralelizable con xargs -P).
# Salida por archivo:
#   "L <clave> <conteo>"  → agregados: lines, benign, malicious, nofam
#   "P <conteo> <familia>" → una línea por familia distinta
# El merge (sort + awk en el pipeline) suma los conteos entre archivos.
{
    # label y family viven en el header JSON (~primeros 400 chars: sha256, tlsh,
    # dates, detection_ratio, label, file_type, family...); el resto de la línea
    # son las ~2568 features numéricas (~19 KB). Acotar el regex al prefijo
    # recorta ~10x el CPU sin tocar los campos buscados. El merge verifica los
    # totales contra los censos oficiales (1,560,000/1,560,000) — si algún
    # campo cayera fuera del prefijo, el total no cuadraría y se detecta.
    head = substr($0, 1, 2048)
    n++
    if (head ~ /"label": *0[^0-9]/) b++
    else if (head ~ /"label": *1[^0-9]/) {
        m++
        if (head ~ /"family": *null/) fn++
        else if (match(head, /"family": *"[^"]*"/)) {
            s = substr(head, RSTART, RLENGTH)
            gsub(/^"family": *"|"$/, "", s)
            fam[s]++
        }
    }
}
END {
    printf "L lines %d\n", n
    printf "L benign %d\n", b + 0
    printf "L malicious %d\n", m + 0
    printf "L nofam %d\n", fn + 0
    for (f in fam) printf "P %d %s\n", fam[f], f
}
