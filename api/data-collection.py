import requests
import csv

def fetch_weather_data(location, date1, date2, api_key):
   
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date1}/{date2}?key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch data. HTTP Status: {response.status_code}"}

def save_to_csv(data, filename="weather_data.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Temperature (°C)", "Humidity (%)", "Cloud Cover (%)", "Conditions"])
        for day in data.get("days", []):
            writer.writerow([
                day.get("datetime", ""),
                day.get("temp", ""),
                day.get("humidity", ""),
                day.get("cloudcover", ""),
                day.get("conditions", "")
            ])

def main():
    api_key = "HFNY98A2AU3QFGWR4X2YWBFPA"
    location = "NewYork" 
    date1 = "2023-11-01" 
    date2 = "2023-11-10" 
    
    weather_data = fetch_weather_data(location, date1, date2, api_key)
    
    if "error" in weather_data:
        print("Error:", weather_data["error"])
    else:
        save_to_csv(weather_data, filename="weather_data.csv")
        print(f"Weather data saved to weather_data.csv")

if __name__ == "__main__":
    main()
