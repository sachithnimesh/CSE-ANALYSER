
import os
import json
import pandas as pd
from glob import glob

from datetime import datetime
from dotenv import load_dotenv, find_dotenv

import os
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import os
import json
import pandas as pd
from collections import defaultdict

# List of symbols
symbols = [
    "ABAN.N0000", "AFSL.N0000", "AEL.N0000", "ACL.N0000", "APLA.N0000", "ACME.N0000", "AGAL.N0000", "AGPL.N0000",
    "AGST.N0000", "AHUN.N0000", "SPEN.N0000", "ALLI.N0000", "AFS.N0000", "ALUM.N0000", "ABL.N0000", "ATL.N0000",
    "TAP.N0000", "GREG.N0000", "AAF.N0000", "ACAP.N0000", "ASIY.N0000", "AHPL.N0000", "ASIR.N0000", "AMSL.N0000",
    "AMF.N0000", "BPPL.N0000", "BFL.N0000", "BALA.N0000", "BRR.N0000", "BERU.N0000", "BOGA.N0000", "BOPL.N0000",
    "BRWN.N0000", "BBH.N0000", "BIL.N0000", "CIC.X0000", "CIC.N0000", "COLO.N0000", "CTHR.N0000", "CTLD.N0000",
    "CWM.N0000", "CSLK.N0000", "CALC.U0000", "CALI.U0000", "CALT.N0000", "CARG.N0000", "CBNK.N0000", "CABO.N0000",
    "CARS.N0000", "CFIN.N0000", "CIND.N0000", "CINS.N0000", "CINS.X0000", "CCS.N0000", "GRAN.N0000", "GUAR.N0000",
    "CHL.N0000", "CHL.X0000", "CHOT.N0000", "KZOO.N0000", "CTBL.N0000", "CTC.N0000", "CHMX.N0000", "LLUB.N0000",
    "CWL.N0000", "CDB.X0000", "REEF.N0000", "COOP.N0000", "PHAR.N0000", "CFI.N0000", "CIT.N0000", "CLND.N0000",
    "COMB.X0000", "COMB.N0000", "COCR.N0000", "COMD.N0000", "SOY.N0000", "DPL.N0000", "DFCC.N0000", "DIAL.N0000",
    "CALF.N0000", "DIMO.N0000", "PKME.N0000", "CTEA.N0000", "DIPD.N0000", "DIST.N0000", "STAF.N0000", "ECL.N0000",
    "EBCR.N0000", "EML.N0000", "EAST.N0000", "EMER.N0000", "EDEN.N0000", "ELPL.N0000", "PACK.N0000", "EXT.N0000",
    "CFVF.N0000", "FCT.N0000", "GHLL.N0000", "WAPO.N0000", "MEL.N0000", "HAPU.N0000", "HARI.N0000", "HNB.X0000",
    "HNB.N0000", "HPL.N0000", "HAYC.N0000", "MGT.N0000", "HEXP.N0000", "CONN.N0000", "HAYL.N0000", "HELA.N0000",
    "HHL.N0000", "CITH.N0000", "HASU.N0000", "HNBF.N0000", "HNBF.X0000", "HOPL.N0000", "HDFC.N0000", "HBS.N0000",
    "HUNA.N0000", "HVA.N0000", "ASPH.N0000", "JINS.N0000", "JAT.N0000", "JETS.N0000", "JKH.N0000", "KHL.N0000",
    "KAHA.N0000", "KPHL.N0000", "KFP.N0000", "KGAL.N0000", "KCAB.N0000", "TYRE.N0000", "KVAL.N0000", "KOTA.N0000",
    "LOLC.N0000", "LVEF.N0000", "LALU.N0000", "ASHO.N0000", "CERA.N0000", "LCBF.N0000", "LIOC.N0000", "LMF.N0000",
    "ASCO.N0000", "TILE.N0000", "LVEN.N0000", "LWL.N0000", "LCEY.N0000", "LDEV.N0000", "LGL.N0000", "LGL.X0000",
    "LPL.N0000", "LPL.X0000", "LITE.N0000", "LFIN.N0000", "LION.N0000", "LOFC.N0000", "LGIL.N0000", "HPFL.N0000",
    "LUMX.N0000", "MADU.N0000", "MFPE.N0000", "MCPL.N0000", "MRH.N0000", "MAL.X0000", "MAL.N0000", "MARA.N0000",
    "MASK.N0000", "MELS.N0000", "MSL.N0000", "MBSL.N0000", "MHDL.N0000", "MULL.N0000", "MDL.N0000", "NAMU.N0000",
    "CSF.N0000", "NDB.N0000", "NTB.N0000", "NHL.N0000", "ODEL.N0000", "BFN.N0000", "OSEA.N0000", "PALM.N0000",
    "PABC.N0000", "PAP.N0000", "PEG.N0000", "PINS.N0000", "PLC.N0000", "GLAS.N0000", "PMB.N0000", "PLR.N0000",
    "CARE.N0000", "RIL.N0000", "RGEM.N0000", "RWSL.N0000", "RAL.N0000", "RENU.N0000", "COCO.N0000", "COCO.X0000",
    "RHL.N0000", "RCH.N0000", "HPWR.N0000", "RICH.N0000", "REXP.N0000", "RCL.N0000", "RPBH.N0000", "SAMP.N0000",
    "SDB.N0000", "SDF.N0000", "SMOT.N0000", "SHOT.X0000", "SHOT.N0000", "SEYB.N0000", "SEYB.X0000", "CSD.N0000",
    "SIRA.N0000", "SIGV.N0000", "SINS.N0000", "SFIN.N0000", "SINH.N0000", "SEMB.N0000", "SEMB.X0000", "SCAP.N0000",
    "CRL.N0000", "SHL.N0000", "AAIC.N0000", "SLTL.N0000", "SUN.N0000", "PARQ.N0000", "TAJ.N0000", "TPL.N0000",
    "TANG.N0000", "TSML.N0000", "TJL.N0000", "TESS.X0000", "TESS.N0000", "LHL.N0000", "CFLB.N0000", "KHC.N0000",
    "SERV.N0000", "LHCL.N0000", "TAFL.N0000", "TKYO.N0000", "TKYO.X0000", "TRAN.N0000", "UBF.N0000", "UAL.N0000",
    "UBC.N0000", "UCAR.N0000", "UML.N0000", "VFIN.N0000", "VONE.N0000", "VPEL.N0000", "VLL.X0000", "VLL.N0000",
    "CITW.N0000", "WATA.N0000", "WIND.N0000"
] 


folder_path = r"D:\Browns\CSE ANALYSER\Local Storage"  # Your actual path
all_dataframes = []

for comp_symbol in symbols:
    file_path = os.path.join(folder_path, f"{comp_symbol}.json")
    
    if not os.path.exists(file_path):
        print(f"{comp_symbol}.json not found. Skipping.")
        continue

    # Load base data
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    if comp_symbol not in data:
        print(f"Key '{comp_symbol}' not in JSON. Skipping.")
        continue

    records = data[comp_symbol]
    df = pd.DataFrame(records)
    df.columns = [col.replace('\ufeff', '') for col in df.columns]
    
    if 'Trade Date' not in df.columns or 'Close (Rs.)' not in df.columns:
        print(f"Missing required columns for {comp_symbol}. Skipping.")
        continue

    df_filtered = df[['Trade Date', 'Close (Rs.)']].copy()
    df_filtered['Trade Date'] = pd.to_datetime(df_filtered['Trade Date'])
    
    # Add daily upload data
    company_data = []
    for filename in os.listdir(folder_path):
        if filename.startswith('20') and filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8-sig') as f:
                document = json.load(f)
            
            upload_date = document.get("upload_date")
            if not upload_date:
                continue

            for company in document.get("data", []):
                if company["Symbol"] != comp_symbol:
                    continue
                last_trade_str = company.get("Last Trade Rs", "0").replace(",", "")
                try:
                    last_trade = float(last_trade_str)
                except ValueError:
                    last_trade = None

                company_data.append({
                    "Trade Date": pd.to_datetime(upload_date, format="%Y%m%d"),
                    "Close (Rs.)": last_trade
                })

    if company_data:
        df_new = pd.DataFrame(company_data)
        df_combined = pd.concat([df_filtered, df_new], ignore_index=True)
    else:
        df_combined = df_filtered

    df_combined.drop_duplicates(subset="Trade Date", keep="last", inplace=True)
    df_combined.sort_values(by="Trade Date", inplace=True)
    df_combined["Symbol"] = comp_symbol  # Add Symbol column
    
    all_dataframes.append(df_combined)

# Combine all into one final DataFrame
final_df = pd.concat(all_dataframes, ignore_index=True)

# Show output
print("\nCombined Final DataFrame:")
print(final_df)

# Save the final DataFrame as a CSV file
output_path = r"D:\Browns\CSE ANALYSER\combined_final_data.csv"
final_df.to_csv(output_path, index=False)
print(f"\nFinal DataFrame saved to {output_path}")