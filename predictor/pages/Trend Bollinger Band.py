from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def get_stock_details(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'][-1]
        company_name = stock.info['longName']
        exchange = stock.info['exchange']
        currency = stock.info['currency']
        volume = stock.history(period='1d')['Volume'][-1]

        return {
            'Ticker': ticker,
            'Company Name': company_name,
            'Current Price': current_price,
            'Exchange': exchange,
            'Currency': currency,
            'Volume': volume,
            
        }
    except Exception as e:
        return "Currently No Data Found!!"

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['UpperBand'] = data['MA'] + (data['Close'].rolling(window=window).std() * num_std_dev)
    data['LowerBand'] = data['MA'] - (data['Close'].rolling(window=window).std() * num_std_dev)
    
    return data

###################################################################################

yf.pdr_override()

st.title('Stock Trend Prediction (Bollinger Bands)')


user_input = st.selectbox('Enter Stock Ticker', (
    "20MICRONS.NS","21STCENMGM.NS", "360ONE.NS", "3IINFOLTD.NS", "3MINDIA.NS", "3PLAND.NS", "5PAISA.NS", "63MOONS.NS", "A2ZINFRA.NS", "AAATECH.NS", "AAKASH.NS", "AAREYDRUGS.NS", "AARON.NS", "AARTECH.NS", "AARTIDRUGS.NS", "AARTIIND.NS", "AARTIPHARM.NS", "AARTISURF.NS", "AARVEEDEN.NS", "AARVI.NS", "AAVAS.NS", "ABAN.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ABMINTLLTD.NS", "ABSLAMC.NS", "ACC.NS", "ACCELYA.NS", "ACCURACY.NS", "ACE.NS", "ACEINTEG.NS", "ACI.NS", "ACL.NS", "ACLGATI.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ADFFOODS.NS", "ADL.NS", "ADORWELD.NS", "ADROITINFO.NS", "ADSL.NS", "ADVANIHOTR.NS", "ADVENZYMES.NS", "AEGISCHEM.NS", "AEROFLEX.NS", "AETHER.NS", "AFFLE.NS", "AGARIND.NS", "AGI.NS", "AGRITECH.NS", "AGROPHOS.NS", "AGSTRA.NS", "AHL.NS", "AHLADA.NS", "AHLEAST.NS", "AHLUCONT.NS", "AIAENG.NS", "AIRAN.NS", "AIROLAM.NS", "AJANTPHARM.NS", "AJMERA.NS", "AJOONI.NS", "AKASH.NS", "AKG.NS", "AKI.NS", "AKSHAR.NS", "AKSHARCHEM.NS", "AKSHOPTFBR.NS", "AKZOINDIA.NS", "ALANKIT.NS", "ALBERTDAVD.NS", "ALEMBICLTD.NS", "ALICON.NS", "ALKALI.NS", "ALKEM.NS", "ALKYLAMINE.NS", "ALLCARGO.NS", "ALLSEC.NS", "ALMONDZ.NS", "ALOKINDS.NS", "ALPA.NS", "ALPHAGEO.NS", "ALPSINDUS.NS", "AMBER.NS", "AMBICAAGAR.NS", "AMBIKCO.NS", "AMBUJACEM.NS", "AMDIND.NS", "AMIORG.NS", "AMJLAND.NS", "AMNPLST.NS", "AMRUTANJAN.NS", "ANANDRATHI.NS", "ANANTRAJ.NS", "ANDHRAPAP.NS", "ANDHRSUGAR.NS", "ANGELONE.NS", "ANIKINDS.NS", "ANKITMETAL.NS", "ANMOL.NS", "ANSALAPI.NS", "ANTGRAPHIC.NS", "ANUP.NS", "ANURAS.NS", "APARINDS.NS", "APCL.NS", "APCOTEXIND.NS", "APEX.NS", "APLAPOLLO.NS", "APLLTD.NS", "APOLLO.NS", "APOLLOHOSP.NS", "APOLLOPIPE.NS", "APOLLOTYRE.NS", "APOLSINHOT.NS", "APTECHT.NS", "APTUS.NS", "ARCHIDPLY.NS", "ARCHIES.NS", "ARE&M.NS", "ARENTERP.NS", "ARIES.NS", "ARIHANTCAP.NS", "ARIHANTSUP.NS", "ARMANFIN.NS", "AROGRANITE.NS", "ARROWGREEN.NS", "ARSHIYA.NS", "ARTEMISMED.NS", "ARTNIRMAN.NS", "ARVEE.NS", "ARVIND.NS", "ARVINDFASN.NS", "ARVSMART.NS", "ASAHIINDIA.NS", "ASAHISONG.NS", "ASAL.NS", "ASALCBR.NS", "ASHAPURMIN.NS", "ASHIANA.NS", "ASHIMASYN.NS", "ASHOKA.NS", "ASHOKAMET.NS", "ASHOKLEY.NS", "ASIANENE.NS", "ASIANHOTNR.NS", "ASIANPAINT.NS", "ASIANTILES.NS", "ASKAUTOLTD.NS", "ASMS.NS", "ASPINWALL.NS", "ASTEC.NS", "ASTERDM.NS", "ASTRAL.NS", "ASTRAMICRO.NS", "ASTRAZEN.NS", "ASTRON.NS", "ATALREAL.NS", "ATAM.NS", "ATFL.NS", "ATGL.NS", "ATL.NS", "ATLANTAA.NS", "ATUL.NS", "ATULAUTO.NS", "AUBANK.NS", "AURIONPRO.NS", "AUROPHARMA.NS", "AURUM.NS", "AUSOMENT.NS", "AUTOAXLES.NS", "AUTOIND.NS", "AVADHSUGAR.NS", "AVALON.NS", "AVANTIFEED.NS", "AVG.NS", "AVONMORE.NS", "AVROIND.NS", "AVTNPL.NS", "AWHCL.NS", "AWL.NS", "AXISBANK.NS", "AXISCADES.NS", "AXITA.NS", "AYMSYNTEX.NS", "AZAD.NS", "BAFNAPH.NS", "BAGFILMS.NS", "BAIDFIN.NS", "BAJAJ-AUTO.NS", "BAJAJCON.NS", "BAJAJELEC.NS", "BAJAJFINSV.NS", "BAJAJHCARE.NS", "BAJAJHIND.NS", "BAJAJHLDNG.NS", "BAJEL.NS", "BAJFINANCE.NS", "BALAJITELE.NS", "BALAMINES.NS", "BALAXI.NS", "BALKRISHNA.NS", "BALKRISIND.NS", "BALMLAWRIE.NS", "BALPHARMA.NS", "BALRAMCHIN.NS", "BANARBEADS.NS", "BANARISUG.NS", "BANCOINDIA.NS", "BANDHANBNK.NS", "BANG.NS", "BANKA.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BANSWRAS.NS", "BARBEQUE.NS", "BASF.NS", "BASML.NS", "BATAINDIA.NS", "BAYERCROP.NS", "BBL.NS", "BBOX.NS", "BBTC.NS", "BBTCL.NS", "BCG.NS", "BCLIND.NS", "BCONCEPTS.NS", "BDL.NS", "BEARDSELL.NS", "BECTORFOOD.NS", "BEDMUTHA.NS", "BEL.NS", "BEML.NS", "BEPL.NS", "BERGEPAINT.NS", "BFINVEST.NS", "BFUTILITIE.NS", "BGRENERGY.NS", "BHAGCHEM.NS", "BHAGERIA.NS", "BHAGYANGR.NS", "BHANDARI.NS", "BHARATFORG.NS", "BHARATGEAR.NS", "BHARATRAS.NS", "BHARATWIRE.NS", "BHARTIARTL.NS", "BHEL.NS", "BIGBLOC.NS", "BIKAJI.NS", "BIL.NS", "BINANIIND.NS", "BIOCON.NS", "BIOFILCHEM.NS", "BIRLACABLE.NS", "BIRLACORPN.NS", "BIRLAMONEY.NS", "BKMINDST.NS", "BLAL.NS", "BLBLIMITED.NS", "BLISSGVS.NS", "BLKASHYAP.NS", "BLS.NS", "BLUECHIP.NS", "BLUECOAST.NS", "BLUEDART.NS", "BLUEJET.NS", "BLUESTARCO.NS", "BODALCHEM.NS", "BOHRAIND.NS", "BOMDYEING.NS", "BOROLTD.NS", "BORORENEW.NS", "BOSCHLTD.NS", "BPCL.NS", "BPL.NS", "BRIGADE.NS", "BRITANNIA.NS", "BRNL.NS", "BROOKS.NS", "BSE.NS", "BSHSL.NS", "BSL.NS", "BSOFT.NS", "BTML.NS", "BURNPUR.NS", "BUTTERFLY.NS", "BVCL.NS", "BYKE.NS", "CALSOFT.NS", "CAMLINFINE.NS", "CAMPUS.NS", "CAMS.NS", "CANBK.NS", "CANFINHOME.NS", "CANTABIL.NS", "CAPACITE.NS", "CAPLIPOINT.NS", "CAPTRUST.NS", "CARBORUNIV.NS", "CAREERP.NS", "CARERATING.NS", "CARTRADE.NS", "CARYSIL.NS", "CASTROLIND.NS", "CCHHL.NS", "CCL.NS", "CDSL.NS", "CEATLTD.NS", "CELEBRITY.NS", "CELLO.NS", "CENTENKA.NS", "CENTEXT.NS", "CENTRALBK.NS", "CENTRUM.NS", "CENTUM.NS", "CENTURYPLY.NS", "CENTURYTEX.NS", "CERA.NS", "CEREBRAINT.NS", "CESC.NS", "CGCL.NS", "CGPOWER.NS", "CHALET.NS", "CHAMBLFERT.NS", "CHEMBOND.NS", "CHEMCON.NS", "CHEMFAB.NS", "CHEMPLASTS.NS", "CHENNPETRO.NS", "CHEVIOT.NS", "CHOICEIN.NS", "CHOLAFIN.NS", "CHOLAHLDNG.NS", "CIEINDIA.NS", "CIGNITITEC.NS", "CINELINE.NS", "CINEVISTA.NS", "CIPLA.NS", "CLEAN.NS", "CLEDUCATE.NS", "CLSEL.NS", "CMSINFO.NS", "COALINDIA.NS", "COASTCORP.NS", "COCHINSHIP.NS", "COFFEEDAY.NS", "COFORGE.NS", "COLPAL.NS", "COMPINFO.NS", "COMPUSOFT.NS", "COMSYN.NS", "CONCOR.NS", "CONCORDBIO.NS", "CONFIPET.NS", "CONSOFINVT.NS", "CONTROLPR.NS", "CORALFINAC.NS", "CORDSCABLE.NS", "COROMANDEL.NS", "COSMOFIRST.NS", "COUNCODOS.NS", "CRAFTSMAN.NS", "CREATIVE.NS", "CREATIVEYE.NS", "CREDITACC.NS", "CREST.NS", "CRISIL.NS", "CROMPTON.NS", "CROWN.NS", "CSBBANK.NS", "CSLFINANCE.NS", "CTE.NS", "CUB.NS", "CUBEXTUB.NS", "CUMMINSIND.NS", "CUPID.NS", "CYBERMEDIA.NS", "CYBERTECH.NS", "CYIENT.NS", "CYIENTDLM.NS", "DABUR.NS", "DALBHARAT.NS", "DALMIASUG.NS", "DAMODARIND.NS", "DANGEE.NS", "DATAMATICS.NS", "DATAPATTNS.NS", "DAVANGERE.NS", "DBCORP.NS", "DBL.NS", "DBOL.NS", "DBREALTY.NS", "DBSTOCKBRO.NS", "DCAL.NS", "DCBBANK.NS", "DCI.NS", "DCM.NS", "DCMFINSERV.NS", "DCMNVL.NS", "DCMSHRIRAM.NS", "DCMSRIND.NS", "DCW.NS", "DCXINDIA.NS", "DECCANCE.NS", "DEEPAKFERT.NS", "DEEPAKNTR.NS", "DEEPENR.NS", "DEEPINDS.NS", "DELHIVERY.NS", "DELPHIFX.NS", "DELTACORP.NS", "DELTAMAGNT.NS", "DEN.NS", "DENORA.NS", "DEVIT.NS", "DEVYANI.NS", "DGCONTENT.NS", "DHAMPURSUG.NS", "DHANBANK.NS", "DHANI.NS", "DHANUKA.NS", "DHARMAJ.NS", "DHRUV.NS", "DHUNINV.NS", "DIACABS.NS", "DIAMINESQ.NS", "DIAMONDYD.NS", "DICIND.NS", "DIGIDRIVE.NS", "DIGISPICE.NS", "DIGJAMLMTD.NS", "DIL.NS", "DISHTV.NS", "DIVGIITTS.NS", "DIVISLAB.NS", "DIXON.NS", "DJML.NS", "DLF.NS", "DLINKINDIA.NS", "DMART.NS", "DMCC.NS", "DNAMEDIA.NS", "DODLA.NS", "DOLATALGO.NS", "DOLLAR.NS", "DOLPHIN.NS", "DOMS.NS", "DONEAR.NS", "DPABHUSHAN.NS", "DPSCLTD.NS", "DPWIRES.NS", "DRCSYSTEMS.NS", "DREAMFOLKS.NS", "DREDGECORP.NS", "DRREDDY.NS", "DSSL.NS", "DTIL.NS", "DUCON.NS", "DVL.NS", "DWARKESH.NS", "DYCL.NS", "DYNAMATECH.NS", "DYNPRO.NS", "E2E.NS", "EASEMYTRIP.NS", "ECLERX.NS", "EDELWEISS.NS", "EDUCOMP.NS", "EICHERMOT.NS", "EIDPARRY.NS", "EIFFL.NS", "EIHAHOTELS.NS", "EIHOTEL.NS", "EIMCOELECO.NS", "EKC.NS", "ELDEHSG.NS", "ELECON.NS", "ELECTCAST.NS", "ELECTHERM.NS", "ELGIEQUIP.NS", "ELGIRUBCO.NS", "ELIN.NS", "EMAMILTD.NS", "EMAMIPAP.NS", "EMAMIREAL.NS", "EMIL.NS", "EMKAY.NS", "EMMBI.NS", "EMSLIMITED.NS", "EMUDHRA.NS", "ENDURANCE.NS", "ENERGYDEV.NS", "ENGINERSIN.NS", "ENIL.NS", "EPACK.NS", "EPIGRAL.NS", "EPL.NS", "EQUIPPP.NS", "EQUITASBNK.NS", "ERIS.NS", "EROSMEDIA.NS", "ESABINDIA.NS", "ESAFSFB.NS", "ESCORTS.NS", "ESSARSHPNG.NS", "ESSENTIA.NS", "ESTER.NS", "ETHOSLTD.NS", "EUROTEXIND.NS", "EVEREADY.NS", "EVERESTIND.NS", "EXCEL.NS", "EXCELINDUS.NS", "EXIDEIND.NS", "EXPLEOSOL.NS", "EXXARO.NS", "FACT.NS", "FAIRCHEMOR.NS", "FAZE3Q.NS", "FCL.NS", "FCONSUMER.NS", "FCSSOFT.NS", "FDC.NS", "FEDERALBNK.NS", "FEDFINA.NS", "FELDVR.NS", "FIBERWEB.NS", "FIEMIND.NS", "FILATEX.NS", "FINCABLES.NS", "FINEORG.NS", "FINOPB.NS", "FINPIPE.NS", "FIVESTAR.NS", "FLAIR.NS", "FLEXITUFF.NS", "FLFL.NS", "FLUOROCHEM.NS", "FMGOETZE.NS", "FMNL.NS", "FOCUS.NS", "FOODSIN.NS", "FORTIS.NS", "FOSECOIND.NS", "FSC.NS", "FSL.NS", "FUSION.NS", "GABRIEL.NS", "GAEL.NS", "GAIL.NS", "GALAXYSURF.NS", "GALLANTT.NS", "GANDHAR.NS", "GANDHITUBE.NS", "GANECOS.NS", "GANESHBE.NS", "GANESHHOUC.NS", "GANGAFORGE.NS", "GANGESSECU.NS", "GARFIBRES.NS", "GATECH.NS", "GATECHDVR.NS", "GATEWAY.NS", "GAYAPROJ.NS", "GEECEE.NS", "GEEKAYWIRE.NS", "GENCON.NS", "GENESYS.NS", "GENSOL.NS", "GENUSPAPER.NS", "GENUSPOWER.NS", "GEOJITFSL.NS", "GEPIL.NS", "GESHIP.NS", "GET&D.NS", "GFLLIMITED.NS", "GHCL.NS", "GHCLTEXTIL.NS", "GICHSGFIN.NS", "GICRE.NS", "GILLANDERS.NS", "GILLETTE.NS", "GINNIFILA.NS", "GIPCL.NS", "GKWLIMITED.NS", "GLAND.NS", "GLAXO.NS", "GLENMARK.NS", "GLFL.NS", "GLOBAL.NS", "GLOBALVECT.NS", "GLOBE.NS", "GLOBUSSPR.NS", "GLS.NS", "GMBREW.NS", "GMDCLTD.NS", "GMMPFAUDLR.NS", "GMRINFRA.NS", "GMRP&UI.NS", "GNA.NS", "GNFC.NS", "GOACARBON.NS", "GOCLCORP.NS", "GOCOLORS.NS", "GODFRYPHLP.NS", "GODHA.NS", "GODREJAGRO.NS", "GODREJCP.NS", "GODREJIND.NS", "GODREJPROP.NS", "GOKEX.NS", "GOKUL.NS", "GOKULAGRO.NS", "GOLDENTOBC.NS", "GOLDIAM.NS", "GOLDTECH.NS", "GOODLUCK.NS", "GOYALALUM.NS", "GPIL.NS", "GPPL.NS", "GPTINFRA.NS", "GRANULES.NS", "GRAPHITE.NS", "GRASIM.NS", "GRAVITA.NS", "GREAVESCOT.NS", "GREENLAM.NS", "GREENPANEL.NS", "GREENPLY.NS", "GREENPOWER.NS", "GRINDWELL.NS", "GRINFRA.NS", "GRMOVER.NS", "GROBTEA.NS", "GRPLTD.NS", "GRSE.NS", "GRWRHITECH.NS", "GSFC.NS", "GSLSU.NS", "GSPL.NS", "GSS.NS", "GTECJAINX.NS", "GTL.NS", "GTLINFRA.NS", "GTPL.NS", "GUFICBIO.NS", "GUJALKALI.NS", "GUJAPOLLO.NS", "GUJGASLTD.NS", "GUJRAFFIA.NS", "GULFOILLUB.NS", "GULFPETRO.NS", "GULPOLY.NS", "GVKPIL.NS", "GVPTECH.NS", "HAL.NS", "HAPPSTMNDS.NS", "HAPPYFORGE.NS", "HARDWYN.NS", "HARIOMPIPE.NS", "HARRMALAYA.NS", "HARSHA.NS", "HATHWAY.NS", "HATSUN.NS", "HAVELLS.NS", "HAVISHA.NS", "HBLPOWER.NS", "HBSL.NS", "HCC.NS", "HCG.NS", "HCL-INSYS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HDIL.NS", "HEADSUP.NS", "HECPROJECT.NS", "HEG.NS", "HEIDELBERG.NS", "HEMIPROP.NS", "HERANBA.NS", "HERCULES.NS", "HERITGFOOD.NS", "HEROMOTOCO.NS", "HESTERBIO.NS", "HEUBACHIND.NS", "HEXATRADEX.NS", "HFCL.NS", "HGINFRA.NS", "HGS.NS", "HIKAL.NS", "HIL.NS", "HILTON.NS", "HIMATSEIDE.NS", "HINDALCO.NS", "HINDCOMPOS.NS", "HINDCON.NS", "HINDCOPPER.NS", "HINDMOTORS.NS", "HINDNATGLS.NS", "HINDOILEXP.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDWAREAP.NS", "HINDZINC.NS", "HIRECT.NS", "HISARMETAL.NS", "HITECH.NS", "HITECHCORP.NS", "HITECHGEAR.NS", "HLEGLAS.NS", "HLVLTD.NS", "HMAAGRO.NS", "HMT.NS", "HMVL.NS", "HNDFDS.NS", "HOMEFIRST.NS", "HONASA.NS", "HONAUT.NS", "HONDAPOWER.NS", "HOVS.NS", "HPAL.NS", "HPIL.NS", "HPL.NS", "HSCL.NS", "HTMEDIA.NS", "HUBTOWN.NS", "HUDCO.NS", "HUHTAMAKI.NS", "HYBRIDFIN.NS", "IBREALEST.NS", "IBULHSGFIN.NS", "ICDSLTD.NS", "ICEMAKE.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "ICIL.NS", "ICRA.NS", "IDBI.NS", "IDEA.NS", "IDEAFORGE.NS", "IDFC.NS", "IDFCFIRSTB.NS", "IEL.NS", "IEX.NS", "IFBAGRO.NS", "IFBIND.NS", "IFCI.NS", "IFGLEXPOR.NS", "IGARASHI.NS", "IGL.NS", "IGPL.NS", "IIFL.NS", "IIFLSEC.NS", "IITL.NS", "IKIO.NS", "IL&FSENGG.NS", "IL&FSTRANS.NS", "IMAGICAA.NS", "IMFA.NS", "IMPAL.NS", "IMPEXFERRO.NS", "INCREDIBLE.NS", "INDBANK.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAGLYCO.NS", "INDIAMART.NS", "INDIANB.NS", "INDIANCARD.NS", "INDIANHUME.NS", "INDIASHLTR.NS", "INDIGO.NS", "INDIGOPNTS.NS", "INDNIPPON.NS", "INDOAMIN.NS", "INDOBORAX.NS", "INDOCO.NS", "INDORAMA.NS", "INDOSTAR.NS", "INDOTECH.NS", "INDOTHAI.NS", "INDOWIND.NS", "INDRAMEDCO.NS", "INDSWFTLAB.NS", "INDSWFTLTD.NS", "INDTERRAIN.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFIBEAM.NS", "INFOBEAN.NS", "INFOMEDIA.NS", "INFY.NS", "INGERRAND.NS", "INNOVACAP.NS", "INOXGREEN.NS", "INOXINDIA.NS", "INOXWIND.NS", "INSECTICID.NS", "INTELLECT.NS", "INTENTECH.NS", "INTLCONV.NS", "INVENTURE.NS", "IOB.NS", "IOC.NS", "IOLCP.NS", "IONEXCHANG.NS", "IPCALAB.NS", "IPL.NS", "IRB.NS", "IRCON.NS", "IRCTC.NS", "IREDA.NS", "IRFC.NS", "IRIS.NS", "IRISDOREME.NS", "IRMENERGY.NS", "ISEC.NS", "ISFT.NS", "ISGEC.NS", "ISMTLTD.NS", "ITC.NS", "ITDC.NS", "ITDCEM.NS", "ITI.NS", "IVC.NS", "IVP.NS", "IWEL.NS", "IZMO.NS", "J&KBANK.NS", "JAGRAN.NS", "JAGSNPHARM.NS", "JAIBALAJI.NS", "JAICORPLTD.NS", "JAIPURKURT.NS", "JAMNAAUTO.NS", "JASH.NS", "JAYAGROGN.NS", "JAYBARMARU.NS", "JAYNECOIND.NS", "JAYSREETEA.NS", "JBCHEPHARM.NS", "JBMA.NS", "JCHAC.NS", "JETAIRWAYS.NS", "JETFREIGHT.NS", "JHS.NS", "JINDALPHOT.NS", "JINDALPOLY.NS", "JINDALSAW.NS", "JINDALSTEL.NS", "JINDRILL.NS", "JINDWORLD.NS", "JIOFIN.NS", "JISLDVREQS.NS", "JISLJALEQS.NS", "JITFINFRA.NS", "JKCEMENT.NS", "JKIL.NS", "JKLAKSHMI.NS", "JKPAPER.NS", "JKTYRE.NS", "JLHL.NS", "JMA.NS", "JMFINANCIL.NS", "JOCIL.NS", "JPASSOCIAT.NS", "JPOLYINVST.NS", "JPPOWER.NS", "JSL.NS", "JSWENERGY.NS", "JSWHL.NS", "JSWINFRA.NS", "JSWSTEEL.NS", "JTEKTINDIA.NS", "JTLIND.NS", "JUBLFOOD.NS", "JUBLINDS.NS", "JUBLINGREA.NS", "JUBLPHARMA.NS", "JUSTDIAL.NS", "JWL.NS", "JYOTHYLAB.NS", "JYOTICNC.NS", "JYOTISTRUC.NS", "KABRAEXTRU.NS", "KAJARIACER.NS", "KAKATCEM.NS", "KALAMANDIR.NS", "KALYANI.NS", "KALYANIFRG.NS", "KALYANKJIL.NS", "KAMATHOTEL.NS", "KAMDHENU.NS", "KAMOPAINTS.NS", "KANANIIND.NS", "KANORICHEM.NS", "KANPRPLA.NS", "KANSAINER.NS", "KAPSTON.NS", "KARMAENG.NS", "KARURVYSYA.NS", "KAVVERITEL.NS", "KAYA.NS", "KAYNES.NS", "KBCGLOBAL.NS", "KCP.NS", "KCPSUGIND.NS", "KDDL.NS", "KEC.NS", "KECL.NS", "KEEPLEARN.NS", "KEI.NS", "KELLTONTEC.NS", "KERNEX.NS", "KESORAMIND.NS", "KEYFINSERV.NS", "KFINTECH.NS", "KHADIM.NS", "KHAICHEM.NS", "KHAITANLTD.NS", "KHANDSE.NS", "KICL.NS", "KILITCH.NS", "KIMS.NS", "KINGFA.NS", "KIOCL.NS", "KIRIINDUS.NS", "KIRLOSBROS.NS", "KIRLOSENG.NS", "KIRLOSIND.NS", "KIRLPNU.NS", "KITEX.NS", "KKCL.NS", "KMSUGAR.NS", "KNRCON.NS", "KOHINOOR.NS", "KOKUYOCMLN.NS", "KOLTEPATIL.NS", "KOPRAN.NS", "KOTAKBANK.NS", "KOTARISUG.NS", "KOTHARIPET.NS", "KOTHARIPRO.NS", "KPIGREEN.NS", "KPIL.NS", "KPITTECH.NS", "KPRMILL.NS", "KRBL.NS", "KREBSBIO.NS", "KRIDHANINF.NS", "KRISHANA.NS", "KRITI.NS", "KRITIKA.NS", "KRITINUT.NS", "KRSNAA.NS", "KSB.NS", "KSCL.NS", "KSHITIJPOL.NS", "KSL.NS", "KSOLVES.NS", "KTKBANK.NS", "KUANTUM.NS", "L&TFH.NS", "LAGNAM.NS", "LAL.NS", "LALPATHLAB.NS", "LAMBODHARA.NS", "LANDMARK.NS", "LAOPALA.NS", "LASA.NS", "LATENTVIEW.NS", "LATTEYS.NS", "LAURUSLABS.NS", "LAXMICOT.NS", "LAXMIMACH.NS", "LCCINFOTEC.NS", "LEMONTREE.NS", "LEXUS.NS", "LFIC.NS", "LGBBROSLTD.NS", "LGBFORGE.NS", "LGHL.NS", "LIBAS.NS", "LIBERTSHOE.NS", "LICHSGFIN.NS", "LICI.NS", "LIKHITHA.NS", "LINC.NS", "LINCOLN.NS", "LINDEINDIA.NS", "LLOYDSENGG.NS", "LLOYDSME.NS", "LODHA.NS", "LOKESHMACH.NS", "LORDSCHLO.NS", "LOTUSEYE.NS", "LOVABLE.NS", "LOYALTEX.NS", "LPDC.NS", "LT.NS", "LTFOODS.NS", "LTIM.NS", "LTTS.NS", "LUMAXIND.NS", "LUMAXTECH.NS", "LUPIN.NS", "LUXIND.NS", "LXCHEM.NS", "LYKALABS.NS", "LYPSAGEMS.NS", "M&M.NS", "M&MFIN.NS", "MAANALU.NS", "MACPOWER.NS", "MADHAV.NS", "MADHUCON.NS", "MADRASFERT.NS", "MAGADSUGAR.NS", "MAGNUM.NS", "MAHABANK.NS", "MAHAPEXLTD.NS", "MAHASTEEL.NS", "MAHEPC.NS", "MAHESHWARI.NS", "MAHLIFE.NS", "MAHLOG.NS", "MAHSCOOTER.NS", "MAHSEAMLES.NS", "MAITHANALL.NS", "MALLCOM.NS", "MALUPAPER.NS", "MANAKALUCO.NS", "MANAKCOAT.NS", "MANAKSIA.NS", "MANAKSTEEL.NS", "MANALIPETC.NS", "MANAPPURAM.NS", "MANGALAM.NS", "MANGCHEFER.NS", "MANGLMCEM.NS", "MANINDS.NS", "MANINFRA.NS", "MANKIND.NS", "MANOMAY.NS", "MANORAMA.NS", "MANORG.NS", "MANUGRAPH.NS", "MANYAVAR.NS", "MAPMYINDIA.NS", "MARALOVER.NS", "MARATHON.NS", "MARICO.NS", "MARINE.NS", "MARKSANS.NS", "MARSHALL.NS", "MARUTI.NS", "MASFIN.NS", "MASTEK.NS", "MATRIMONY.NS", "MAWANASUG.NS", "MAXESTATES.NS", "MAXHEALTH.NS", "MAXIND.NS", "MAYURUNIQ.NS", "MAZDA.NS", "MAZDOCK.NS", "MBAPL.NS", "MBECL.NS", "MBLINFRA.NS", "MCDOWELL-N.NS", "MCL.NS", "MCLEODRUSS.NS", "MCX.NS", "MEDANTA.NS", "MEDIASSIST.NS", "MEDICAMEQ.NS", "MEDICO.NS", "MEDPLUS.NS", "MEGASOFT.NS", "MEGASTAR.NS", "MELSTAR.NS", "MENONBE.NS", "MEP.NS", "METROBRAND.NS", "METROPOLIS.NS", "MFSL.NS", "MGEL.NS", "MGL.NS", "MHLXMIRU.NS", "MHRIL.NS", "MICEL.NS", "MIDHANI.NS", "MINDACORP.NS", "MINDTECK.NS", "MIRCELECTR.NS", "MIRZAINT.NS", "MITCON.NS", "MITTAL.NS", "MKPL.NS", "MMFL.NS", "MMP.NS", "MMTC.NS", "MODIRUBBER.NS", "MODISONLTD.NS", "MODTHREAD.NS", "MOHITIND.NS", "MOIL.NS", "MOKSH.NS", "MOL.NS", "MOLDTECH.NS", "MOLDTKPAC.NS", "MONARCH.NS", "MONTECARLO.NS", "MORARJEE.NS", "MOREPENLAB.NS", "MOTHERSON.NS", "MOTILALOFS.NS", "MOTISONS.NS", "MOTOGENFIN.NS", "MPHASIS.NS", "MPSLTD.NS", "MRF.NS", "MRO-TEK.NS", "MRPL.NS", "MSPL.NS", "MSTCLTD.NS", "MSUMI.NS", "MTARTECH.NS", "MTEDUCARE.NS", "MTNL.NS", "MUFIN.NS", "MUFTI.NS", "MUKANDLTD.NS", "MUKTAARTS.NS", "MUNJALAU.NS", "MUNJALSHOW.NS", "MURUDCERA.NS", "MUTHOOTCAP.NS", "MUTHOOTFIN.NS", "MUTHOOTMF.NS", "MVGJL.NS", "NACLIND.NS", "NAGAFERT.NS", "NAGREEKCAP.NS", "NAGREEKEXP.NS", "NAHARCAP.NS", "NAHARINDUS.NS", "NAHARPOLY.NS", "NAHARSPING.NS", "NAM-INDIA.NS", "NARMADA.NS", "NATCOPHARM.NS", "NATHBIOGEN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NAVA.NS", "NAVINFLUOR.NS", "NAVKARCORP.NS", "NAVNETEDUL.NS", "NAZARA.NS", "NBCC.NS", "NBIFIN.NS", "NCC.NS", "NCLIND.NS", "NDGL.NS", "NDL.NS", "NDLVENTURE.NS", "NDRAUTO.NS", "NDTV.NS", "NECCLTD.NS", "NECLIFE.NS", "NELCAST.NS", "NELCO.NS", "NEOGEN.NS", "NESCO.NS", "NESTLEIND.NS", "NETWEB.NS", "NETWORK18.NS", "NEULANDLAB.NS", "NEWGEN.NS", "NEXTMEDIA.NS", "NFL.NS", "NGIL.NS", "NGLFINE.NS", "NH.NS", "NHPC.NS", "NIACL.NS", "NIBL.NS", "NIITLTD.NS", "NIITMTS.NS", "NILAINFRA.NS", "NILASPACES.NS", "NILKAMAL.NS", "NINSYS.NS", "NIPPOBATRY.NS", "NIRAJ.NS", "NIRAJISPAT.NS", "NITCO.NS", "NITINSPIN.NS", "NITIRAJ.NS", "NKIND.NS", "NLCINDIA.NS", "NMDC.NS", "NOCIL.NS", "NOIDATOLL.NS", "NORBTEAEXP.NS", "NOVAAGRI.NS", "NRAIL.NS", "NRBBEARING.NS", "NRL.NS", "NSIL.NS", "NSLNISP.NS", "NTPC.NS", "NUCLEUS.NS", "NURECA.NS", "NUVAMA.NS", "NUVOCO.NS", "NYKAA.NS", "OAL.NS", "OBCL.NS", "OBEROIRLTY.NS", "OCCL.NS", "OFSS.NS", "OIL.NS", "OILCOUNTUB.NS", "OLECTRA.NS", "OMAXAUTO.NS", "OMAXE.NS", "OMINFRAL.NS", "OMKARCHEM.NS", "ONELIFECAP.NS", "ONEPOINT.NS", "ONGC.NS", "ONMOBILE.NS", "ONWARDTEC.NS", "OPTIEMUS.NS", "ORBTEXP.NS", "ORCHPHARMA.NS", "ORICONENT.NS", "ORIENTALTL.NS", "ORIENTBELL.NS", "ORIENTCEM.NS", "ORIENTCER.NS", "ORIENTELEC.NS", "ORIENTHOT.NS", "ORIENTLTD.NS", "ORIENTPPR.NS", "ORISSAMINE.NS", "ORTEL.NS", "ORTINLAB.NS", "OSIAHYPER.NS", "OSWALAGRO.NS", "OSWALGREEN.NS", "OSWALSEEDS.NS", "PAGEIND.NS", "PAISALO.NS", "PAKKA.NS", "PALASHSECU.NS", "PALREDTEC.NS", "PANACEABIO.NS", "PANACHE.NS", "PANAMAPET.NS", "PANSARI.NS", "PAR.NS", "PARACABLES.NS", "PARADEEP.NS", "PARAGMILK.NS", "PARAS.NS", "PARASPETRO.NS", "PARSVNATH.NS", "PASUPTAC.NS", "PATANJALI.NS", "PATELENG.NS", "PATINTLOG.NS", "PAVNAIND.NS", "PAYTM.NS", "PCBL.NS", "PCJEWELLER.NS", "PDMJEPAPER.NS", "PDSL.NS", "PEARLPOLY.NS", "PEL.NS", "PENIND.NS", "PENINLAND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PFIZER.NS", "PFOCUS.NS", "PFS.NS", "PGEL.NS", "PGHH.NS", "PGHL.NS", "PGIL.NS", "PHOENIXLTD.NS", "PIDILITIND.NS", "PIGL.NS", "PIIND.NS", "PILANIINVS.NS", "PILITA.NS", "PIONEEREMB.NS", "PITTIENG.NS", "PIXTRANS.NS", "PKTEA.NS", "PLASTIBLEN.NS", "PLAZACABLE.NS", "PNB.NS", "PNBGILTS.NS", "PNBHOUSING.NS", "PNC.NS", "PNCINFRA.NS", "POCL.NS", "PODDARHOUS.NS", "PODDARMENT.NS", "POKARNA.NS", "POLICYBZR.NS", "POLYCAB.NS", "POLYMED.NS", "POLYPLEX.NS", "PONNIERODE.NS", "POONAWALLA.NS", "POWERGRID.NS", "POWERINDIA.NS", "POWERMECH.NS", "PPAP.NS", "PPL.NS", "PPLPHARMA.NS", "PRAENG.NS", "PRAJIND.NS", "PRAKASH.NS", "PRAKASHSTL.NS", "PRAXIS.NS", "PRECAM.NS", "PRECOT.NS", "PRECWIRE.NS", "PREMEXPLN.NS", "PREMIERPOL.NS", "PRESTIGE.NS", "PRICOLLTD.NS", "PRIMESECU.NS", "PRINCEPIPE.NS", "PRITI.NS", "PRITIKAUTO.NS", "PRIVISCL.NS", "PROZONER.NS", "PRSMJOHNSN.NS", "PRUDENT.NS", "PRUDMOULI.NS", "PSB.NS", "PSPPROJECT.NS", "PTC.NS", "PTCIL.NS", "PTL.NS", "PUNJABCHEM.NS", "PURVA.NS", "PVP.NS", "PVRINOX.NS", "PYRAMID.NS", "QUESS.NS", "QUICKHEAL.NS", "RACE.NS", "RADAAN.NS", "RADHIKAJWE.NS", "RADIANTCMS.NS", "RADICO.NS", "RADIOCITY.NS", "RAILTEL.NS", "RAIN.NS", "RAINBOW.NS", "RAJESHEXPO.NS", "RAJMET.NS", "RAJRATAN.NS", "RAJRILTD.NS", "RAJSREESUG.NS", "RAJTV.NS", "RALLIS.NS", "RAMANEWS.NS", "RAMAPHO.NS", "RAMASTEEL.NS", "RAMCOCEM.NS", "RAMCOIND.NS", "RAMCOSYS.NS", "RAMKY.NS", "RAMRAT.NS", "RANASUG.NS", "RANEENGINE.NS", "RANEHOLDIN.NS", "RATEGAIN.NS", "RATNAMANI.NS", "RATNAVEER.NS", "RAYMOND.NS", "RBA.NS", "RBL.NS", "RBLBANK.NS", "RBZJEWEL.NS", "RCF.NS", "RCOM.NS", "RECLTD.NS", "REDINGTON.NS", "REDTAPE.NS", "REFEX.NS", "REGENCERAM.NS", "RELAXO.NS", "RELCHEMQ.NS", "RELIANCE.NS", "RELIGARE.NS", "RELINFRA.NS", "REMSONSIND.NS", "RENUKA.NS", "REPCOHOME.NS", "REPL.NS", "REPRO.NS", "RESPONIND.NS", "RGL.NS", "RHFL.NS", "RHIM.NS", "RHL.NS", "RICOAUTO.NS", "RIIL.NS", "RISHABH.NS", "RITCO.NS", "RITES.NS", "RKDL.NS", "RKEC.NS", "RKFORGE.NS", "RML.NS", "ROHLTD.NS", "ROLEXRINGS.NS", "ROLLT.NS", "ROLTA.NS", "ROML.NS", "ROSSARI.NS", "ROSSELLIND.NS", "ROTO.NS", "ROUTE.NS", "RPGLIFE.NS", "RPOWER.NS", "RPPINFRA.NS", "RPPL.NS", "RPSGVENT.NS", "RRKABEL.NS", "RSSOFTWARE.NS", "RSWM.NS", "RSYSTEMS.NS", "RTNINDIA.NS", "RTNPOWER.NS", "RUBYMILLS.NS", "RUCHINFRA.NS", "RUCHIRA.NS", "RUPA.NS", "RUSHIL.NS", "RUSTOMJEE.NS", "RVHL.NS", "RVNL.NS", "S&SPOWER.NS", "SABEVENTS.NS", "SADBHAV.NS", "SADBHIN.NS", "SADHNANIQ.NS", "SAFARI.NS", "SAGARDEEP.NS", "SAGCEM.NS", "SAH.NS", "SAHYADRI.NS", "SAIL.NS", "SAKAR.NS", "SAKHTISUG.NS", "SAKSOFT.NS", "SAKUMA.NS", "SALASAR.NS", "SALONA.NS", "SALSTEEL.NS", "SALZERELEC.NS", "SAMBHAAV.NS", "SAMHI.NS", "SAMPANN.NS", "SANDESH.NS", "SANDHAR.NS", "SANDUMA.NS", "SANGAMIND.NS", "SANGHIIND.NS", "SANGHVIMOV.NS", "SANGINITA.NS", "SANOFI.NS", "SANSERA.NS", "SANWARIA.NS", "SAPPHIRE.NS", "SARDAEN.NS", "SAREGAMA.NS", "SARLAPOLY.NS", "SARVESHWAR.NS", "SASKEN.NS", "SASTASUNDR.NS", "SATIA.NS", "SATIN.NS", "SATINDLTD.NS", "SBC.NS", "SBCL.NS", "SBFC.NS", "SBGLP.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SCHAEFFLER.NS", "SCHAND.NS", "SCHNEIDER.NS", "SCI.NS", "SCPL.NS", "SDBL.NS", "SEAMECLTD.NS", "SECMARK.NS", "SECURCRED.NS", "SECURKLOUD.NS", "SELAN.NS", "SELMC.NS", "SEMAC.NS", "SENCO.NS", "SEPC.NS", "SEQUENT.NS", "SERVOTECH.NS", "SESHAPAPER.NS", "SETCO.NS", "SETUINFRA.NS", "SEYAIND.NS", "SFL.NS", "SGIL.NS", "SGL.NS", "SHAH.NS", "SHAHALLOYS.NS", "SHAILY.NS", "SHAKTIPUMP.NS", "SHALBY.NS", "SHALPAINTS.NS", "SHANKARA.NS", "SHANTI.NS", "SHANTIGEAR.NS", "SHARDACROP.NS", "SHARDAMOTR.NS", "SHAREINDIA.NS", "SHEMAROO.NS", "SHILPAMED.NS", "SHIVALIK.NS", "SHIVAMAUTO.NS", "SHIVAMILLS.NS", "SHIVATEX.NS", "SHK.NS", "SHOPERSTOP.NS", "SHRADHA.NS", "SHREDIGCEM.NS", "SHREECEM.NS", "SHREEPUSHK.NS", "SHREERAMA.NS", "SHRENIK.NS", "SHREYANIND.NS", "SHREYAS.NS", "SHRIPISTON.NS", "SHRIRAMFIN.NS", "SHRIRAMPPS.NS", "SHYAMCENT.NS", "SHYAMMETL.NS", "SHYAMTEL.NS", "SICALLOG.NS", "SIEMENS.NS", "SIGACHI.NS", "SIGIND.NS", "SIGMA.NS", "SIGNATURE.NS", "SIKKO.NS", "SIL.NS", "SILGO.NS", "SILINV.NS", "SILLYMONKS.NS", "SILVERTUC.NS", "SIMBHALS.NS", "SIMPLEXINF.NS", "SINDHUTRAD.NS", "SINTERCOM.NS", "SIRCA.NS", "SIS.NS", "SIYSIL.NS", "SJS.NS", "SJVN.NS", "SKFINDIA.NS", "SKIPPER.NS", "SKIPPER-RE.NS", "SKMEGGPROD.NS", "SKYGOLD.NS", "SMARTLINK.NS", "SMCGLOBAL.NS", "SMLISUZU.NS", "SMLT.NS", "SMSLIFE.NS", "SMSPHARMA.NS", "SNOWMAN.NS", "SOBHA.NS", "SOFTTECH.NS", "SOLARA.NS", "SOLARINDS.NS", "SOMANYCERA.NS", "SOMATEX.NS", "SOMICONVEY.NS", "SONACOMS.NS", "SONAMLTD.NS", "SONATSOFTW.NS", "SOTL.NS", "SOUTHBANK.NS", "SOUTHWEST.NS", "SPAL.NS", "SPANDANA.NS", "SPARC.NS", "SPCENET.NS", "SPECIALITY.NS", "SPENCERS.NS", "SPIC.NS", "SPLIL.NS", "SPLPETRO.NS", "SPMLINFRA.NS", "SPORTKING.NS", "SPYL.NS", "SREEL.NS", "SRF.NS", "SRGHFL.NS", "SRHHYPOLTD.NS", "SRPL.NS", "SSWL.NS", "STAR.NS", "STARCEMENT.NS", "STARHEALTH.NS", "STARPAPER.NS", "STARTECK.NS", "STCINDIA.NS", "STEELCAS.NS", "STEELCITY.NS", "STEELXIND.NS", "STEL.NS", "STERTOOLS.NS", "STLTECH.NS", "STOVEKRAFT.NS", "STYLAMIND.NS", "STYRENIX.NS", "SUBEXLTD.NS", "SUBROS.NS", "SUDARSCHEM.NS", "SUKHJITS.NS", "SULA.NS", "SUMEETINDS.NS", "SUMICHEM.NS", "SUMIT.NS", "SUMMITSEC.NS", "SUNCLAY.NS", "SUNDARAM.NS", "SUNDARMFIN.NS", "SUNDARMHLD.NS", "SUNDRMBRAK.NS", "SUNDRMFAST.NS", "SUNFLAG.NS", "SUNPHARMA.NS", "SUNTECK.NS", "SUNTV.NS", "SUPERHOUSE.NS", "SUPERSPIN.NS", "SUPRAJIT.NS", "SUPREMEENG.NS", "SUPREMEIND.NS", "SUPREMEINF.NS", "SUPRIYA.NS", "SURAJEST.NS", "SURANASOL.NS", "SURANAT&P.NS", "SURYALAXMI.NS", "SURYAROSNI.NS", "SURYODAY.NS", "SUTLEJTEX.NS", "SUULD.NS", "SUVEN.NS", "SUVENPHAR.NS", "SUVIDHAA.NS", "SUZLON.NS", "SVLL.NS", "SVPGLOB.NS", "SWANENERGY.NS", "SWARAJENG.NS", "SWELECTES.NS", "SWSOLAR.NS", "SYMPHONY.NS", "SYNCOMF.NS", "SYNGENE.NS", "SYRMA.NS", "TAINWALCHM.NS", "TAJGVK.NS", "TAKE.NS", "TALBROAUTO.NS", "TANLA.NS", "TARAPUR.NS", "TARC.NS", "TARMAT.NS", "TARSONS.NS", "TASTYBITE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAINVEST.NS", "TATAMETALI.NS", "TATAMOTORS.NS", "TATAMTRDVR.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TATATECH.NS", "TATVA.NS", "TBZ.NS", "TCI.NS", "TCIEXP.NS", "TCLCONS.NS", "TCNSBRANDS.NS", "TCPLPACK.NS", "TCS.NS", "TDPOWERSYS.NS", "TEAMLEASE.NS", "TECHIN.NS", "TECHM.NS", "TECHNOE.NS", "TECILCHEM.NS", "TEGA.NS", "TEJASNET.NS", "TEMBO.NS", "TERASOFT.NS", "TEXINFRA.NS", "TEXMOPIPES.NS", "TEXRAIL.NS", "TFCILTD.NS", "TFL.NS", "TGBHOTELS.NS", "THANGAMAYL.NS", "THEINVEST.NS", "THEJO.NS", "THEMISMED.NS", "THERMAX.NS", "THOMASCOOK.NS", "THOMASCOTT.NS", "THYROCARE.NS", "TI.NS", "TIDEWATER.NS", "TIIL.NS", "TIINDIA.NS", "TIJARIA.NS", "TIL.NS", "TIMESGTY.NS", "TIMETECHNO.NS", "TIMKEN.NS", "TIPSFILMS.NS", "TIPSINDLTD.NS", "TIRUMALCHM.NS", "TIRUPATIFL.NS", "TITAGARH.NS", "TITAN.NS", "TMB.NS", "TNPETRO.NS", "TNPL.NS", "TNTELE.NS", "TOKYOPLAST.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TOTAL.NS", "TOUCHWOOD.NS", "TPHQ.NS", "TPLPLASTEH.NS", "TRACXN.NS", "TREEHOUSE.NS", "TREJHARA.NS", "TREL.NS", "TRENT.NS", "TRF.NS", "TRIDENT.NS", "TRIGYN.NS", "TRIL.NS", "TRITURBINE.NS", "TRIVENI.NS", "TRU.NS", "TTKHLTCARE.NS", "TTKPRESTIG.NS", "TTL.NS", "TTML.NS", "TV18BRDCST.NS", "TVSELECT.NS", "TVSHLTD.NS", "TVSMOTOR.NS", "TVSSCS.NS", "TVSSRICHAK.NS", "TVTODAY.NS", "TVVISION.NS", "UBL.NS", "UCAL.NS", "UCOBANK.NS", "UDS.NS", "UFLEX.NS", "UFO.NS", "UGARSUGAR.NS", "UGROCAP.NS", "UJJIVAN.NS", "UJJIVANSFB.NS", "ULTRACEMCO.NS", "UMAEXPORTS.NS", "UMANGDAIRY.NS", "UMESLTD.NS", "UNICHEMLAB.NS", "UNIDT.NS", "UNIENTER.NS", "UNIINFO.NS", "UNIONBANK.NS", "UNIPARTS.NS", "UNITECH.NS", "UNITEDPOLY.NS", "UNITEDTEA.NS", "UNIVASTU.NS", "UNIVCABLES.NS", "UNIVPHOTO.NS", "UNOMINDA.NS", "UPL.NS", "URAVI.NS", "URJA.NS", "USHAMART.NS", "USK.NS", "UTIAMC.NS", "UTKARSHBNK.NS", "UTTAMSUGAR.NS", "V2RETAIL.NS", "VADILALIND.NS", "VAIBHAVGBL.NS", "VAISHALI.NS", "VAKRANGEE.NS", "VALIANTLAB.NS", "VALIANTORG.NS", "VARDHACRLC.NS", "VARDMNPOLY.NS", "VARROC.NS", "VASCONEQ.NS", "VASWANI.NS", "VBL.NS", "VCL.NS", "VEDL.NS", "VENKEYS.NS", "VENUSPIPES.NS", "VENUSREM.NS", "VERANDA.NS", "VERTOZ.NS", "VESUVIUS.NS", "VETO.NS", "VGUARD.NS", "VHL.NS", "VIDHIING.NS", "VIJAYA.NS", "VIJIFIN.NS", "VIKASECO.NS", "VIKASLIFE.NS", "VIMTALABS.NS", "VINATIORGA.NS", "VINDHYATEL.NS", "VINEETLAB.NS", "VINNY.NS", "VINYLINDIA.NS", "VIPCLOTHNG.NS", "VIPIND.NS", "VIPULLTD.NS", "VIRINCHI.NS", "VISAKAIND.NS", "VISASTEEL.NS", "VISESHINFO.NS", "VISHNU.NS", "VISHWARAJ.NS", "VIVIDHA.NS", "VLEGOV.NS", "VLSFINANCE.NS", "VMART.NS", "VOLTAMP.NS", "VOLTAS.NS", "VPRPL.NS", "VRLLOG.NS", "VSSL.NS", "VSTIND.NS", "VSTTILLERS.NS", "VTL.NS", "WABAG.NS", "WALCHANNAG.NS", "WANBURY.NS", "WEALTH.NS", "WEBELSOLAR.NS", "WEIZMANIND.NS", "WEL.NS", "WELCORP.NS", "WELENT.NS", "WELINV.NS", "WELSPUNLIV.NS", "WENDT.NS", "WESTLIFE.NS", "WEWIN.NS", "WHEELS.NS", "WHIRLPOOL.NS", "WILLAMAGOR.NS", "WINDLAS.NS", "WINDMACHIN.NS", "WINSOME.NS", "WIPL.NS", "WIPRO.NS", "WOCKPHARMA.NS", "WONDERLA.NS", "WORTH.NS", "WSI.NS", "WSTCSTPAPR.NS", "XCHANGING.NS", "XELPMOC.NS", "XPROINDIA.NS", "YAARI.NS", "YASHO.NS", "YATHARTH.NS", "YATRA.NS", "YESBANK.NS", "YUKEN.NS", "ZAGGLE.NS", "ZEEL.NS", "ZEELEARN.NS", "ZEEMEDIA.NS", "ZENITHEXPO.NS", "ZENITHSTL.NS", "ZENSARTECH.NS", "ZENTEC.NS", "ZFCVINDIA.NS", "ZIMLAB.NS", "ZODIAC.NS", "ZODIACLOTH.NS", "ZOMATO.NS", "ZOTA.NS", "ZUARI.NS", "ZUARIIND.NS", "ZYDUSLIFE.NS", "ZYDUSWELL.NS",
    # Nifty Indices
    "^NSEI", "^NIFTYBANK", "^NIFTYIT", "^NIFTYPHARMA", "^NIFTYFMCG", "^NIFTYAUTO", "^NIFTYMETAL", "^NIFTYENERGY", "^NIFTYINFRA",
    "^NIFTYREALTY", "^NIFTYMID50", "^NIFTYCPSE", "^NIFTYCOMMODITIES", "^NIFTYCONSUMPTION", "^NIFTYPSE", "^NIFTYPSUBANK", "^NIFTYFINSERVICE",
    "^NIFTY50DIV", "^NIFTY100", "^NIFTY200", "^NIFTY500", "^NIFTYDIVOPPS50", "^NIFTYDIV30", "^NIFTYHIGHBETA50", "^NIFTYLOWVOL50", 
    "^NIFTYQUALITY30", "^NIFTY500VALUE50", "^NIFTYMIDCAP150FREE", "^NIFTY50TMC", "^NIFTY100TMC", "^NIFTY200TMC", "^NIFTY500TMC", 
    "^NIFTY50EW", "^NIFTY100EW", "^NIFTY200EW", "^NIFTY500EW", "^NIFTY100LIQ15", "^NIFTYMIDCAP50", "^NIFTYMIDSML400", "^NIFTYMIDSML150", 
    "^NIFTY50DIVCAP", "^NIFTY500DIV", "^NIFTYALPHA50", "^NIFTY100ALPHA", "^NIFTY200ALPHA", "^NIFTYCPSEINDEX", "^NIFTYPVTBANK", 
    "^NIFTYPSUBANKINDEX", "^NIFTYMIDSMALLCAP400", "^NIFTYMIDCAP50", "^NIFTYSMEEMERGE", "^NIFTYCOMMODITIESSMALLCAP250", 
    "^NIFTYBANKNIFTYFINANCIALSERVICESINDEX", "^NIFTYBANKNIFTYFINANCIALSERVICES60", "^NIFTYBANKNIFTYFINANCIALSERVICES20", 
    "^NIFTYBANKNIFTYICICIBANK", "^NIFTYBANKNIFTYHDFCBANK", "^NIFTYBANKNIFTYPRIVATEBANK", "^NIFTYBANKNIFTYPUBLICBANK", 
    "^NIFTYBANKNIFTYPSUBANK", "^NIFTYBANKNIFTYALLPSUBANK", "^NIFTYBANKNIFTYALLFINANCE", "^NIFTYBANKNIFTYALLFINANCE30", 
    "^NIFTYBANKNIFTYALLFINANCE20", "^NIFTYBANKNIFTYALLFINANCE15", "^NIFTYBANKNIFTYALLFINANCE10", "^NIFTYBANKNIFTYINFRA", 
    "^NIFTYBANKNIFTYPSU", "^NIFTYBANKNIFTYFINANCE", "^NIFTYBANKNIFTYALLOFSERVICE", 
    "^NIFTYBANKNIFTYALLOFSERVICE30"
    # Add more stock symbols here...
    ))
# Button to trigger data retrieval
if st.button('Retrieve Data'):
    # Get today's date
    today_date = datetime.now().date()
    # download dataframe
    data = pdr.get_data_yahoo(user_input, start="2020-01-01", end=today_date)

    # Fetch stock details
    stock_details = get_stock_details(user_input)

    if isinstance(stock_details, dict):
        st.subheader(f"Stock Details for {user_input}:")

        # Create columns for displaying data side by side
        col1, col2 = st.columns(2)

        # Display details in the first column
        with col1:
            # st.write("ATTRIBUTE",text_color='red')
            st.markdown(f'<p style="color:orange;">ATTRIBUTE</p>', unsafe_allow_html=True)
            for key in stock_details.keys():
                st.write(key)

        # Display values in the second column
        with col2:
            st.markdown(f'<p style="color:orange;">VALUE</p>', unsafe_allow_html=True)
            for value in stock_details.values():
                st.write(value)
    else:
        st.write(stock_details)

    # Describing Data
    st.subheader('Data From 2020 - 2024')

    # Add more spacing to the descriptive statistics table
    data_description = data.describe().T.style.set_table_styles([
        {'selector': 'tr:hover',
         'props': [('background-color', '#ffffb3')]},
        {'selector': 'th',
         'props': [('background-color', '#e6e6e6')]},
        {'selector': 'td',
         'props': [('border', '2px solid #cccccc')]},
        {'selector': 'th:hover',
         'props': [('background-color', '#ffffb3')]},
        {'selector': 'tr:nth-child(even)',
         'props': [('background-color', '#f2f2f2')]},
    ])

    st.write(data_description)

    # Calculate Bollinger Bands
    data = calculate_bollinger_bands(data)

    # Visualizations
    st.subheader("Closing Price vs Time Chart with Bollinger Bands")
    
    # Create a Plotly figure
    fig_bollinger = go.Figure()

    # Plot closing price
    


    # Plot moving average
    fig_bollinger.add_trace(go.Scatter(
        x=data.index,
        y=data['MA'],
        mode='lines',
        name='Moving Average',
        line=dict(color='orange')
    ))

    # Plot Bollinger Bands
    fig_bollinger.add_trace(go.Scatter(
        x=data.index,
        y=data['UpperBand'],
        mode='lines',
        fill=None,
        opacity=0.2,
        line=dict(color='purple'),
        name='Upper Bollinger Band'
    ))

    fig_bollinger.add_trace(go.Scatter(
        x=data.index,
        y=data['LowerBand'],
        mode='lines',
        fill='tonexty',
        opacity=0.2,
        line=dict(color='purple'),
        name='Lower Bollinger Band'
    ))

    # Update layout
    fig_bollinger.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        title='Closing Price with Moving Average and Bollinger Bands',
        showlegend=True
    )
    
    fig_bollinger.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    ))
    fig_bollinger.update_layout(xaxis_rangeslider_visible=False)
    
    fig_bollinger.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))

    # Display the Plotly chart
    st.plotly_chart(fig_bollinger)

    # Splitting data into training and testing
    data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70): int(len(data))])

    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # Load my model
    model = load_model('keras_model.h5')

    # Testing Part
    past_100_days = data_training.tail(100)
    final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_data)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 25 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader("Prediction vs Original")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Original Trend', line=dict(color='blue'))) 
    fig2.add_trace(go.Scatter(x=data.index[-len(y_predicted):], y=y_predicted[:, 0], mode='lines', name='Predicted Trend', line=dict(color='red')))

    fig2.update_layout(xaxis_title="Time", yaxis_title="Price")
    fig2.update_layout(
    xaxis=dict(
        title='Date',
        showgrid=True  # Disable grid lines for x-axis
    ),
    yaxis=dict(
        title='Price',
        showgrid=True  # Disable grid lines for y-axis
    ))
    st.plotly_chart(fig2)
