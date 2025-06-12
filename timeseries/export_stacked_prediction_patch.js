//Imports

//S1
//S2
//pred_roi



var S1 = ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterDate('2023-10-01', '2024-04-30')
                  .filterBounds(pred_roi)
                  // .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  // Filter to get images collected in interferometric wide swath mode.
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                  .map(function(image){return image.clip(pred_roi)})
                  .select('VV', 'VH');
print(S1,"S1- rabi- prediction patch");  

// mosaic images of the same date
function mosaicByDate(imcol){
  var imlist = imcol.toList(imcol.size());

  var unique_dates = imlist.map(function(im){
    return ee.Image(im).date().format("YYYY-MM-dd");
  }).distinct();

  var mosaic_imlist = unique_dates.map(function(d){
    d = ee.Date(d);

    var im = imcol
      .filterDate(d, d.advance(1, "day"))
      .mosaic();

    return im.set(
        "system:time_start", d.millis(), 
        "system:id", d.format("YYYY-MM-dd"), 
        "system:index", d.format("YYYYMMdd"));
  });

  return ee.ImageCollection(mosaic_imlist);
}
var S1_Mosaic = mosaicByDate(S1);
print(S1_Mosaic, 'S1_Mosaic');


// select VV 
var S1_VV = S1_Mosaic.select('VV'); 
print(S1_VV, 'S1 VV');

// select VH 
var S1_VH = S1_Mosaic.select('VH'); 
print(S1_VH, 'S1 VH');

// convert image collection to image
var VV = S1_VV.toBands();
print('Collection to bands S1_VV',  VV);
var VH = S1_VH.toBands();
print('Collection to bands S1_VH',  VH);

// stack VV and VH images together 
var S1_stack = VV.addBands(VH); 
print(S1_stack, 'S1_stack'); 


// visualise prediction patch 
Map.addLayer(pred_roi, {color:'cyan'}, 'prediction Patch: Ujjain');



// // If the export has more than 1e8 pixels, set "maxPixels" higher.
// Export.image.toDrive({
//   image: S1_stack,
//   description: 'S1_stack',
//   folder: 'crop',
//   region: pred_roi,
//   scale: 10,
//   crs: 'EPSG:4326',
//   maxPixels: 1e13
// });



// Sentinel 2 

var start = '2023-10-15';
var end   = '2024-04-30';

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') 
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50))
                  // .map(maskS2clouds)
                  .filterBounds(pred_roi)
                  .map(function(image){return image.clip(pred_roi)});
print(S2,"S2 Rabi"); 

var S2_Mosaic = mosaicByDate(S2);
print(S2_Mosaic, 'S2_Mosaic');


// ndvi calculation function
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

var s2_ndvi = S2_Mosaic.map(addNDVI);
s2_ndvi     = s2_ndvi.select('NDVI'); 
print(s2_ndvi, 's2_ndvi');


var temporalCollection = function(collection, start, count, interval, units) {
  // Create a sequence of numbers, one for each time interval.
  var sequence = ee.List.sequence(0, ee.Number(count).subtract(1));

  var originalStartDate = ee.Date(start);

  return ee.ImageCollection(sequence.map(function(i) {
    // Get the start date of the current sequence.
    var startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units);

    // Get the end date of the current sequence.
    var endDate = originalStartDate.advance(
      ee.Number(interval).multiply(ee.Number(i).add(1)), units);

    return collection.filterDate(startDate, endDate).min()
        .set('system:time_start', startDate.millis())
        .set('system:time_end', endDate.millis())
        .set( "system:id", startDate.format("YYYY-MM-dd"))
        .set(  "system:index", startDate.format("YYYY-MM-dd"));
  }));
};
var ndvi_16days = temporalCollection(s2_ndvi, start, 13, 16, 'day');
print(ndvi_16days);


var ndvi_16days_size = ndvi_16days.size(); 
print(ndvi_16days_size, 'ndvi_12days_size'); 

var ndvi_16days_list = ndvi_16days.toList(ndvi_16days_size);
print(ndvi_16days_list, 'ndvi_16days_list'); 


var indexToRemove = 1;  // Replace with the index you want to remove
var updated_ndvi_16days_list = ndvi_16days_list.slice(0, indexToRemove)
    .cat(ndvi_16days_list.slice(indexToRemove + 1));

print(updated_ndvi_16days_list, 'Updated NDVI 16 Days List');


// var ndvi_col = updated_ndvi_16days_list.toCollection();
// print(ndvi_col);

// Convert the list to an ImageCollection
var ndvi_col= ee.ImageCollection(updated_ndvi_16days_list);
print(ndvi_col);

var NDVI = ndvi_col.toBands();
print('Collection to bands ndvi',  NDVI);

// stack VV and VH images together 
var stack = VV.addBands(VH); 
stack = stack.addBands(NDVI);
print(stack, 'stack'); 

// Iexport VV VH NDVI
Export.image.toDrive({
  image: stack.toDouble(),
  description: 'stack',
  folder: 'crop',
  region: pred_roi,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
