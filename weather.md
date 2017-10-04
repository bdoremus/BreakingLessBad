# Getting stations near the center of Business Units
```python
df = pd.read_csv('./data/Asset_to_Business_Unit_Translator.csv')
df[['Latitude', 'Longitude']][df.businessUnit == 'DURANGO'].mean()
```
Get Lat/Lon information, then go to:  
https://api.weather.gov/points/LAT,LON/stations

Get station, then go to:
https://www.ncdc.noaa.gov/cdo-web/search


## Farmington
```
Latitude      36.790173
Longitude   -107.802882
```
```
{
    "id": "https://api.weather.gov/stations/KFMN",
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -108.22917,
            36.74361
        ]
    },
    "properties": {
        "@id": "https://api.weather.gov/stations/KFMN",
        "@type": "wx:ObservationStation",
        "elevation": {
            "value": 1677.0096,
            "unitCode": "unit:m"
        },
        "stationIdentifier": "KFMN",
        "name": "Farmington, Four Corners Regional Airport",
        "timeZone": "America/Denver"
    }
},
```

## Wamsutter
```
Latitude      41.620662
Longitude   -107.945351
```
```
{
    "id": "https://api.weather.gov/stations/KRWL",
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -107.19972,
            41.80556
        ]
    },
    "properties": {
        "@id": "https://api.weather.gov/stations/KRWL",
        "@type": "wx:ObservationStation",
        "elevation": {
            "value": 2076.9072,
            "unitCode": "unit:m"
        },
        "stationIdentifier": "KRWL",
        "name": "Rawlins, Rawlins Municipal Airport",
        "timeZone": "America/Denver"
    }
},
```

## Arkoma
```
Latitude     34.955536
Longitude   -95.305227
```
```
{
    "id": "https://api.weather.gov/stations/KMLC",
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -95.78306,
            34.88222
        ]
    },
    "properties": {
        "@id": "https://api.weather.gov/stations/KMLC",
        "@type": "wx:ObservationStation",
        "elevation": {
            "value": 234.0864,
            "unitCode": "unit:m"
        },
        "stationIdentifier": "KMLC",
        "name": "McAlester, McAlester Regional Airport",
        "timeZone": "America/Chicago"
    }
},
```


## Anadarko
```
Latitude      36.042583
Longitude   -100.395370
```
```
{
    "id": "https://api.weather.gov/stations/KHHF",
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -100.4,
            35.9
        ]
    },
    "properties": {
        "@id": "https://api.weather.gov/stations/KHHF",
        "@type": "wx:ObservationStation",
        "elevation": {
            "value": 729.996,
            "unitCode": "unit:m"
        },
        "stationIdentifier": "KHHF",
        "name": "Canadian, Hemphill County Airport",
        "timeZone": "America/Chicago"
    }
},
```

## East Texas
```
Latitude     32.347673
Longitude   -94.440447
```
```
{
    "id": "https://api.weather.gov/stations/KASL",
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -94.3078,
            32.5205
        ]
    },
    "properties": {
        "@id": "https://api.weather.gov/stations/KASL",
        "@type": "wx:ObservationStation",
        "elevation": {
            "value": 108.8136,
            "unitCode": "unit:m"
        },
        "stationIdentifier": "KASL",
        "name": "Marshall",
        "timeZone": "America/Chicago"
    }
}
```

## Durango
```
Latitude      37.137820
Longitude   -107.712583
```
```
{
    "id": "https://api.weather.gov/stations/KDRO",
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [
            -107.76023,
            37.14312
        ]
    },
    "properties": {
        "@id": "https://api.weather.gov/stations/KDRO",
        "@type": "wx:ObservationStation",
        "elevation": {
            "value": 2019.9096,
            "unitCode": "unit:m"
        },
        "stationIdentifier": "KDRO",
        "name": "Durango-La Plata County Airport",
        "timeZone": "America/Denver"
    }
}
```
