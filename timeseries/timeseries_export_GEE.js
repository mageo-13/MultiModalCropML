// Imports 

// S1 ImageCollection 
// S2 ImageCollection 
// ujjain district shapefile (ujjain)
// gt points for crops (ujjain_gt_crop_inc)



var roi = ujjain;

Map.setOptions('satellite');
Map.centerObject(roi, 9);



// visualise roi polygon
Map.addLayer(ujjain, {color:'blue'}, 'ROI: Ujjain');
// visualise prediction patch 
Map.addLayer(prediction_patch_ujjain, {color:'cyan'}, 'prediction Patch: Ujjain');

// // visualise gt points 
// Map.addLayer(ujjain_crop_gt, {color:'yellow'}, 'GT: Crops Ujjain');
// // visualise lulc gt points 
// Map.addLayer(ujjain_lulc_gt, {color:'red'}, 'GT: LULC Ujjain');
// visualise gt points 
Map.addLayer(ujjain_crop_gt_inc, {color:'yellow'}, 'GT: Crops Ujjain Increased');



// Prediction Patch
// print(prediction_patch_ujjain, 'prediction_patch');

// var pred_patch = ee.FeatureCollection(prediction_patch_ujjain);
// print(pred_patch, 'pred_patch feature col');

// // Export the image sample feature collection to Drive as a shapefile.
// Export.table.toDrive({
//   collection: pred_patch,
//   description: 'prediction_patch_ujjain_shp',
//   folder:'crop_eda_localcopy/shapefiles',
//   fileFormat: 'SHP'
// });



//Sentinel 1 : filter images using date and roi --------------------------------------------------
{

// S1 processsing
{ 
var start = '2023-10-01';
var end   = '2024-04-30';

var S1 = ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterDate(start, end)
                  .filterBounds(roi)
                  // .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  // Filter to get images collected in interferometric wide swath mode.
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                  .map(function(image){return image.clip(roi)})
                  .select('VV', 'VH');
print(S1,"S1- rabi");  


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


// var S1_VV = S1desc.select('VV'); 
// print(S1_VV, 'S1 VV');

// select VV 
var S1_VV = S1_Mosaic.select('VV'); 
print(S1_VV, 'S1 VV');

// select VH 
var S1_VH = S1_Mosaic.select('VH'); 
print(S1_VH, 'S1 VH');

// convert image collection to image
// var VV = S1_VV.toBands();
// print('Collection to bands S1_VV',  VV);

// stack VV and VH images together 
// var S1_stack = VV.addBands(VH); 
// print(S1_stack, 'S1_stack'); 

// var first = S1_VV.first();
// var visualizationS1 = {"opacity":1,"bands":["VV","VV","VV"],"min":-20,"max":2,"gamma":10};
// Map.addLayer(first,visualizationS1,"first VV");

// var first = S1_VH.first();
// var visualizationS1 = {"opacity":1,"bands":["VH","VH","VH"],"min":-20,"max":2,"gamma":10};
// Map.addLayer(first,visualizationS1,"first VH");
}



// change according to the band ----------------------------------------------------------------

var collection = S1_VV; 
// var collection = S1_VH;
print(collection, 'collection');

// var points = ujjain_crop_gt; 
// var points = ujjain_lulc_gt; 
var points = ujjain_crop_gt_inc; 




// // We need a unique id for each point. We take the feature id and set it as
// // a property so we can refer to each point easily
// var points = points.map(function(feature) {
//   return ee.Feature(feature.geometry(), {'id': feature.id()});
// }); 




// 1. functions to extract date and time series values - VV ---------------------------------------

{
  
var triplets = collection.map(function(image) {
  return image.select('VV').reduceRegions({
    collection: points, 
    reducer: ee.Reducer.mean().setOutputs(['VV']), 
    scale: 10,
  })// reduceRegion doesn't return any output if the image doesn't intersect
    // with the point or if the image is masked out due to cloud
    // If there was no ndvi value found, we set the ndvi to a NoData value -9999
    .map(function(feature) {
    var vv = ee.List([feature.get('VV'), -9999])
      .reduce(ee.Reducer.firstNonNull());
    return feature.set({'VV': vv, 'imageID': image.id()});
    });
  }).flatten();

print(triplets, 'triplets');



var format = function(table, rowId, colId) {
  var rows = table.distinct(rowId); 
  var joined = ee.Join.saveAll('matches').apply({
    primary: rows, 
    secondary: table, 
    condition: ee.Filter.equals({
      leftField: rowId, 
      rightField: rowId
    })
  });
        
  return joined.map(function(row) {
      var values = ee.List(row.get('matches'))
        .map(function(feature) {
          feature = ee.Feature(feature);
          return [feature.get(colId), feature.get('VV')];
        });
      return row.select([rowId]).set(ee.Dictionary(values.flatten()));
    });
};


// // The result is a 'tall' table. We can further process it to 
// // extract the date from the imageID property.
var tripletsWithDate = triplets.map(function(f) {
  var imageID = f.get('imageID');
  // var date = ee.String(imageID).slice(17,25);
  var date = ee.String(imageID);
  return f.set('date', date);
});

print(tripletsWithDate, 'tripletsWithDate');



// For a cleaner table, we can also filter out
// null values, remove duplicates and sort the table
// before exporting.
// {
// var tripletsFiltered = tripletsWithDate
//   .filter(ee.Filter.neq('VV', -9999))
//   .distinct(['id', 'date'])
//   .sort('id');
// }

// print(tripletsFiltered , 'tripletsFiltered');


// Specify the columns that we want to export -VV
Export.table.toDrive({
    collection: tripletsWithDate,
    // collection: formattedTriplets,
    description: 'VV_timeseries_ujjain_crops3inc',  // change name
    folder: 'Crop',
    fileNamePrefix: 'VV_timeseries_ujjain_crops3inc',  // change name
    fileFormat: 'CSV',
    selectors: ['id', 'date', 'VV', 'crpname_eg', 'lat', 'lon', 'geometry']
    // selectors: ['id', 'date', 'VV', 'crop_name', 'latitude', 'longitude', 'geometry']
    // selectors: ['id', 'date', 'VV', 'LULC_class', 'latitude', 'longitude', 'geometry']
    // selectors: ['id', 'date', 'VV', 'land_use', 'crop_name', 'crpname_eg', 'latitude', 'longitude', 'geometry']
});

}

var collection = S1_VH; 

// 2. functions to extract date and time series values - VH  -------------------------------------------

{
  
var triplets = collection.map(function(image) {
  return image.select('VH').reduceRegions({
    collection: points, 
    reducer: ee.Reducer.mean().setOutputs(['VH']), 
    scale: 10,
  })// reduceRegion doesn't return any output if the image doesn't intersect
    // with the point or if the image is masked out due to cloud
    // If there was no ndvi value found, we set the ndvi to a NoData value -9999
    .map(function(feature) {
    var vh = ee.List([feature.get('VH'), -9999])
      .reduce(ee.Reducer.firstNonNull())
    return feature.set({'VH': vh, 'imageID': image.id()})
    })
  }).flatten();

print(triplets, 'triplets');



var format = function(table, rowId, colId) {
  var rows = table.distinct(rowId); 
  var joined = ee.Join.saveAll('matches').apply({
    primary: rows, 
    secondary: table, 
    condition: ee.Filter.equals({
      leftField: rowId, 
      rightField: rowId
    })
  });
        
  return joined.map(function(row) {
      var values = ee.List(row.get('matches'))
        .map(function(feature) {
          feature = ee.Feature(feature);
          return [feature.get(colId), feature.get('VH')];
        });
      return row.select([rowId]).set(ee.Dictionary(values.flatten()));
    });
};


// // The result is a 'tall' table. We can further process it to 
// // extract the date from the imageID property.
var tripletsWithDate = triplets.map(function(f) {
  var imageID = f.get('imageID');
  // var date = ee.String(imageID).slice(17,25);
  var date = ee.String(imageID);
  return f.set('date', date)
})

print(tripletsWithDate, 'tripletsWithDate');


// We can export this tall table.

// For a cleaner table, we can also filter out
// null values, remove duplicates and sort the table
// before exporting.
// {
// var tripletsFiltered = tripletsWithDate
//   .filter(ee.Filter.neq('VV', -9999))
//   .distinct(['id', 'date'])
//   .sort('id');
// }

// print(tripletsFiltered , 'tripletsFiltered');


// Specify the columns that we want to export - VH
Export.table.toDrive({
    collection: tripletsWithDate,
    // collection: formattedTriplets,
    description: 'VH_timeseries_ujjain_crops3inc',  //change name 
    folder: 'Crop',
    fileNamePrefix: 'VH_timeseries_ujjain_crops3inc',  //change name 
    fileFormat: 'CSV',
    selectors: ['id', 'date', 'VH', 'crpname_eg', 'lat', 'lon', 'geometry']
    // selectors: ['id', 'date', 'VH', 'crop_name', 'latitude', 'longitude', 'geometry']
    // selectors: ['id', 'date', 'VH', 'LULC_class', 'latitude', 'longitude', 'geometry']
    // selectors:['id', 'date', 'VH', 'land_use', 'crop_name', 'crpname_eg', 'latitude', 'longitude', 'geometry']
});
  
}


// // print only first 5 elements
// var tripletsWithDate5  = tripletsWithDate.limit(5);
// print('tripletsWithDate5:',tripletsWithDate5);

}



// Sentinel 2 : filter images using date and roi --------------------------------------------------------------
{

// S2 processsing
{  
var start = '2023-10-15';
var end   = '2024-04-30';
  
/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */
function maskS2clouds(image) {
  var qa =image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000)
  .copyProperties(image, image.propertyNames());
}


var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') 
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50))
                  // .map(maskS2clouds)
                  .filterBounds(roi)
                  .map(function(image){return image.clip(roi)});
print(S2,"S2 Rabi"); 


var visualizationS2 = {
  min: 0.0,
  max: 3000,
  bands: ['B4', 'B3', 'B2'],
};
var visualizationNdvi = {
  min: -1,
  max: 1,
  bands: ['NDVI'],
  palette:['red','yellow','green']
};

// var first = S2.first();
// Map.addLayer(first, visualizationS2, 'S2 first');

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

    return collection.filterDate(startDate, endDate).max()
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

// visualise ndvi images 
ndvi_16days_size.evaluate(function(size) {
  for (var i = 0; i < size; i++) {
    var img = ndvi_16days_list.get(i);
    var imgLayer = ee.Image(img);
    Map.addLayer(imgLayer, visualizationNdvi, 'S2_ndvi16days ' + i);
  }
});

}

// change according to the band ----------------------------------------------------------------

var collection = ndvi_16days; 
print(collection, 'collection');

// var points = ujjain_crop_gt; 




// 1. functions to extract date and time series values - NDVI ---------------------------------------
{
  
var triplets = collection.map(function(image) {
  return image.select('NDVI').reduceRegions({
    collection: points, 
    reducer: ee.Reducer.mean().setOutputs(['NDVI']), 
    scale: 10,
  })// reduceRegion doesn't return any output if the image doesn't intersect
    // with the point or if the image is masked out due to cloud
    // If there was no ndvi value found, we set the ndvi to a NoData value -9999
    .map(function(feature) {
    var ndvi = ee.List([feature.get('NDVI'), -9999])
      .reduce(ee.Reducer.firstNonNull());
    return feature.set({'NDVI': ndvi, 'imageID': image.id()});
    });
  }).flatten();

print(triplets, 'triplets');



var format = function(table, rowId, colId) {
  var rows = table.distinct(rowId); 
  var joined = ee.Join.saveAll('matches').apply({
    primary: rows, 
    secondary: table, 
    condition: ee.Filter.equals({
      leftField: rowId, 
      rightField: rowId
    })
  });
        
  return joined.map(function(row) {
      var values = ee.List(row.get('matches'))
        .map(function(feature) {
          feature = ee.Feature(feature);
          return [feature.get(colId), feature.get('NDVI')];
        });
      return row.select([rowId]).set(ee.Dictionary(values.flatten()));
    });
};


// // The result is a 'tall' table. We can further process it to 
// // extract the date from the imageID property.
var tripletsWithDate = triplets.map(function(f) {
  var imageID = f.get('imageID');
  // var date = ee.String(imageID).slice(17,25);
  var date = ee.String(imageID);
  return f.set('date', date);
});

print(tripletsWithDate, 'tripletsWithDate');



// For a cleaner table, we can also filter out
// null values, remove duplicates and sort the table
// before exporting.
// {
// var tripletsFiltered = tripletsWithDate
//   .filter(ee.Filter.neq('VV', -9999))
//   .distinct(['id', 'date'])
//   .sort('id');
// }

// print(tripletsFiltered , 'tripletsFiltered');


// Specify the columns that we want to export -VV
Export.table.toDrive({
    collection: tripletsWithDate,
    // collection: formattedTriplets,
    description: 'NDVI_timeseries_ujjain_crops3inc',  // change name
    folder: 'Crop',
    fileNamePrefix: 'NDVI_timeseries_ujjain_crops3inc',  // change name
    fileFormat: 'CSV',
    selectors: ['id', 'date', 'NDVI', 'crpname_eg', 'lat', 'lon', 'geometry']
    // selectors: ['id', 'date', 'NDVI', 'land_use', 'crop_name', 'crpname_eg', 'latitude', 'longitude', 'geometry']
    // selectors: ['id', 'date', 'NDVI', 'LULC_class', 'latitude', 'longitude', 'geometry']
});

}


}


