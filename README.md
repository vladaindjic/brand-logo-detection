# Brand Logo Detection and Recognition
Team student project that can be used to detect logo in the picture and recognise its brand. 
In order to do this, it uses two approaches:
- Traditional one based on sliding window method, HOG transformation and neural networks
- Modern one based on Deep Learning that uses Convolutional Neural Networks and selective search.

This projects also makes a comparison between aforementioned approaches. The reasons why the second approach is better than the first one can be found in the document [poster-logo-detection.pdf](https://github.com/vladaindjic/brand-logo-detection/blob/master/poster-logo-detection.pdf)

**Note**: The rest of this document, so as [poster-logo-detection.pdf](https://github.com/vladaindjic/brand-logo-detection/blob/master/poster-logo-detection.pdf), are written in Serbian language for the purposes of the course [Soft Computing](http://www.ftn.uns.ac.rs/n672719865/soft-kompjuting). Both can be easily translated by using Google Translate.


# Detekcija logoa sa slike i određivanje njegovog brenda (Flickr-27)

## Članovi tima 
- SW 1/2014, Žarko Drageljević
- SW 56/2014, Milan Desančić
- SW 4/2014, Vladimir Inđić

# Definicija problema
Prepoznavanje brenda čiji je logo prikazan na slici. 

# Motivacija
Koliko puta nam se desilo da vidimo prodavnicu, piće, hranu ili komad odeće, a da ne znamo koji je brend u pitanju. Zar ne bi bilo korisno ukoliko bismo samo mogli da slikamo predmet i na osnovu logoa prepoznamo pripadajući brend.

# Konceptualno rešenje
Neophodno je pripremiti obiman skup podataka koji u sebi sadrži slike na kojima su prikazani logoi brendova, kao i slike koje nemaju nijedan logo. Sa slika prethodno pomenutog skupa podataka primenom HOG transformacije izdvajamo deskriptor koji koristimo kao ulaz u klasifikator. Izlaz klasifikatora predstavljaće brend loga sa slike ili će se ustanoviti da na slici nema logoa.

# Implementacija rešenja
## Priprema skupa podataka
U projektnom rešenju biće korišćen [Flickr Logos 27 dataset](http://image.ntua.gr/iva/datasets/flickr_logos/). U njemu se nalazi slike sledećih 27 brendova: 
- Adidas 
- Apple
- BMW,
- Citroen, 
- Coca Cola, 
- DHL
- FedEx
- Ferrari
- Ford
- Google
- Heineken
- HP
- Intel
- McDonalds
- Mini
- Nbc
- Nike
- Pepsi 
- Porsche
- Puma
- Red Bull
- Sprite
- Starbucks
- Texaco 
- Unicef
- Vodafone
- Yahoo
- Takođe se nalazi i određeni broj slika na kojima nije prikazan nijedan logo. 
S obzirom na to da je skup podataka relativno mali za obučavanje klasifikatora, potrebno je izvršiti augmentaciju slika (rotiranje, isecanje, dodavanje šuma, blur, ...) iz skupa, kako bi se skup podataka proširio. Skup podataka se deli na skup podataka za treniranje (80% ukupnog skupa podataka) i skup podataka za testiranje (20%).

## HOG transformacija 
Potrebno je implementirati HOG transformaciju i primeniti je na svaku od slika skupa za treniranje, kako bismo izdvojili njen HOG deskriptor.

## Treniranje klasifikatora 
Kao klasifikator koristićemo veštkačku neuronsku mrežu. Njen ulaz predstavlja prethodno određeni HOG deskriptor slike iz trening skupa podataka. Izlaz je predikcija brenda čiji je logo prikazan na slici ili konstatacija da logoa nema.

## Evaluacija dobijenog rešenja 
Na slikama iz skupa za testiranje primenjujemo sledeći postupak: 
- primena HOG transformacije i određivanje deskriptora slike
- dovođenje deskriptora na ulaz neuronske mreže
- predikcija klase kojoj logo pripada
- poređenje sa stvarnom vrednošću.
Očekuje se uspešnost od preko 90%.

# Primena projektnog rešenja 
Na slikama sa društvenih mreža (npr. Instagram, Facebook, Flickr, ...) pokušaćemo da odredimo regione na kojima se nalazi logo i da odredimo njegov brend. Regione ćemo odrediti tako što ćemo primeniti metodu [šetajućeg prozora](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/) kao pretproces za HOG transofmaciju.

## Asistent: 
- @vdragan1993 Dragan Vidaković


### Napomena: 
[Deklinacija reči logo](https://sh.wiktionary.org/wiki/logo)
