# Typing style:
# 1. camelCase: punukUntaKayakGini
# 2. snake_case: semua_kalimat_kecil_tapi_pake_understand
# 3. CapitalCase: BasicallySemuaAwalKataKapital
# 4. Capital_Snake: Fuck_You


# 1. Jangan mulai nama variable dari angka
# 2. jangan pake simbol selain _
# kotakA = "Cilo"

# print(kotakA)


nama = "Cilo, Momon"
print(nama)

beratCilo = 10
beratMomon = 5
print(beratCilo, beratMomon)


# gabungin text
# Opsi 1
print("Cilo beratnya:", beratCilo)
# Opsi 2
print(f"Cilo beratnya: {beratCilo}")


# opsi 1
print("Cilo itu beratnya ", beratCilo, "tapi kalo momon beratnya", beratMomon)
# opsi 2
print(f"Cilo itu beratnya {beratCilo} tapi kalo momon beratnya {beratMomon}")

print("Cilo dan Momon kalau digendong", beratCilo + beratMomon)
print(f"{nama} kalau digendong beratnya {beratCilo + beratMomon}")