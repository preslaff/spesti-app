# CheckAndCompare Prices 📱💰

A sophisticated AI-powered price verification system for Bulgarian consumers to compare BGN and EUR prices on product labels using computer vision and machine learning.

## 🌟 Features

### 📸 **Smart Price Scanning**
- **Real-time OCR**: Advanced Google Vision API with Bulgarian language optimization
- **AI Product Recognition**: GPT-4 powered intelligent product name extraction
- **Spatial Analysis**: Identifies product names vs. prices using advanced algorithms
- **Error Correction**: Handles common OCR mistakes (Latin-to-Cyrillic conversion)

### 🏪 **Store Detection**
- **GPS-based Recognition**: Automatically detects nearby stores using Google Places API
- **Manual Selection**: Dropdown with major Bulgarian chains (Kaufland, Lidl, Metro, Billa, Fantastico, etc.)
- **Smart Fallbacks**: Store-specific coordinate mapping when GPS is unavailable

### 💱 **Price Verification**
- **Real-time Conversion**: BGN to EUR price validation using official exchange rates
- **Visual Feedback**: Color-coded results (green for fair prices, red for overpriced)
- **Historical Tracking**: Price change monitoring over time
- **Accuracy Detection**: Identifies suspicious pricing patterns

### 🗺️ **Interactive Mapping**
- **Live Price Map**: Real-time visualization of price data across Sofia
- **Store Locations**: Accurate mapping of scanning locations
- **Price Trends**: Visual representation of good vs. bad deals
- **Historical Data**: Track price changes geographically

### 📊 **Data Analytics**
- **Google Sheets Integration**: Automatic data logging and storage
- **Product Dictionary**: 1000+ Bulgarian product database with fuzzy matching
- **Price History**: Compare current prices with historical data
- **Trend Analysis**: Identify pricing patterns by store and product

## 🛠️ Technologies Used

### Backend
- **Python Flask** - Web framework
- **Google Cloud Vision API** - OCR processing
- **OpenAI GPT-4** - AI-powered text analysis
- **Google Sheets API** - Data storage
- **Google Maps/Places API** - Location services
- **RapidFuzz** - Fuzzy string matching
- **Pandas** - Data processing

### Frontend
- **HTML5/CSS3/JavaScript** - User interface
- **Google Maps JavaScript API** - Interactive mapping
- **Camera API** - Image capture
- **Responsive Design** - Mobile-optimized interface

## 📋 Prerequisites

### API Keys Required
1. **Google Cloud Console**: Vision API, Sheets API, Maps API, Places API
2. **OpenAI API Key**: For GPT-4 access
3. **Google Sheets**: For data storage

### System Requirements
- Python 3.8+
- Modern web browser with camera access
- Internet connection for API calls

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd checkandcompare-prices
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Google Cloud Services
1. Create a Google Cloud Project
2. Enable APIs: Vision, Sheets, Maps, Places
3. Create Service Account and download `gcp_credentials.json`
4. Place `gcp_credentials.json` in project root

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Required API Keys
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
OPENAI_API_KEY=your_openai_api_key
SPREADSHEET_ID=your_google_sheets_id
BGN_TO_EUR_RATE=1.95583
```

### 5. Set Up Google Sheet
1. Create a new Google Sheet
2. Add these column headers:
   ```
   Timestamp | Product Name | Store Name | BGN Price | EUR Price | Expected EUR | Difference | Status | Latitude | Longitude
   ```
3. Share the sheet with your service account email
4. Copy the Sheet ID to your `.env` file

### 6. Prepare Product Database
Ensure `final_product_name.csv` exists in the root directory with Bulgarian product names.

## 🏃‍♂️ Running the Application

### Start Backend Server
```bash
cd backend
python app.py
```
The Flask server will start on `http://127.0.0.1:5000`

### Access Frontend
Open `frontend/index.html` in your web browser or serve it through a web server.

## 📡 API Endpoints

### Core Endpoints
- `POST /api/verify-prices` - Main price verification endpoint
- `GET /api/prices` - Get price data for map visualization
- `GET /api/stores` - Get available store options
- `GET /api/maps-key` - Get Google Maps API key for frontend

### Utility Endpoints
- `POST /api/setup-sheet` - Initialize Google Sheet headers
- `GET /api/health` - Health check endpoint

### Example Request
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('store', selectedStore);
formData.append('latitude', userLatitude);
formData.append('longitude', userLongitude);

fetch('/api/verify-prices', {
    method: 'POST',
    body: formData
});
```

## 🗂️ Project Structure

```
checkandcompare-prices/
├── backend/
│   ├── app.py                 # Main Flask application
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── index.html            # Main web interface
│   └── server.py             # Optional frontend server
├── gcp_credentials.json      # Google Cloud service account
├── final_product_name.csv    # Bulgarian product database
├── .env                      # Environment variables
└── README.md                 # This file
```

## 🔧 Configuration

### Store Coordinate Mapping
The system includes predefined coordinates for major Bulgarian store chains:

```python
STORE_DEFAULT_COORDINATES = {
    'Kaufland': {'lat': 42.6864, 'lng': 23.3238},
    'Lidl': {'lat': 42.6506, 'lng': 23.3806},
    'Metro': {'lat': 42.6234, 'lng': 23.3844},
    'Billa': {'lat': 42.6977, 'lng': 23.3219},
    'Fantastico': {'lat': 42.6584, 'lng': 23.3486},
    # ... more stores
}
```

### Price Validation Logic
```python
# Current BGN to EUR conversion rate
BGN_TO_EUR_RATE = 1.95583

# Price difference threshold for warnings
PRICE_THRESHOLD = 0.01  # 1 cent tolerance
```

## 🎯 How It Works

### 1. **Image Capture**
- User captures price label using smartphone camera
- System validates image quality and distance

### 2. **OCR Processing**
- Google Vision API extracts Bulgarian text
- Advanced spatial analysis identifies text blocks
- AI corrects common OCR errors

### 3. **Product Recognition**
- GPT-4 analyzes extracted text
- Fuzzy matching against Bulgarian product database
- Eliminates promotional text and noise

### 4. **Store Detection**
- GPS-based automatic detection using Google Places API
- Manual selection from dropdown menu
- Smart fallback coordinates for each store chain

### 5. **Price Verification**
- Identifies BGN and EUR prices from visual prominence
- Validates conversion rate accuracy
- Flags suspicious pricing patterns

### 6. **Data Storage & Visualization**
- Saves results to Google Sheets
- Displays real-time results on interactive map
- Tracks historical price changes

## 🔍 Troubleshooting

### Common Issues

**OCR Not Working**
- Ensure camera permissions are granted
- Check lighting conditions
- Verify image is not blurry or too far

**Store Detection Failed**
- Enable GPS/location services
- Check Google Places API quota
- Manually select store from dropdown

**Price Verification Errors**
- Ensure label has both BGN and EUR prices
- Check that prices are clearly visible
- Verify label is not damaged or obscured

**API Errors**
- Check API key validity and quotas
- Verify internet connection
- Review Google Cloud Console for errors

### Debug Mode
Enable debug logging by setting `debug=True` in `app.run()`:
```python
if __name__ == '__main__':
    app.run(debug=True)
```

## 🛡️ Security Considerations

- API keys stored in environment variables
- Service account credentials properly secured
- Input validation on all user data
- Rate limiting on API endpoints

## 📊 Performance

- **OCR Processing**: ~2-3 seconds per image
- **AI Analysis**: ~1-2 seconds per request
- **Map Loading**: Real-time with 100 recent entries
- **Database**: Google Sheets with unlimited storage

## 🌍 Language Support

- **Primary**: Bulgarian (Cyrillic script)
- **OCR Optimization**: Handles Latin-to-Cyrillic conversion
- **Product Database**: 1000+ Bulgarian product names
- **UI**: Bulgarian language interface

## 📈 Future Enhancements

- [ ] Mobile app development (React Native/Flutter)
- [ ] Machine learning price prediction
- [ ] Push notifications for price alerts
- [ ] Multi-city expansion beyond Sofia
- [ ] Integration with major retailer APIs
- [ ] Blockchain-based price verification
- [ ] Social features (user reviews, ratings)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Cloud Vision API for OCR capabilities
- OpenAI for GPT-4 AI analysis
- Bulgarian consumer advocacy groups
- SoftUni community for testing and feedback

## 📞 Support

For issues, questions, or contributions:
- Create an issue in the GitHub repository
- Contact the development team
- Join our community discussions

---

**Built with ❤️ for Bulgarian consumers by the CheckAndCompare team**