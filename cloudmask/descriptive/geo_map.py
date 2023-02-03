import folium
from folium.plugins import HeatMap
import os

import sentinelhub
from sentinelhub import CRS
from shapely.geometry import shape, Point
from rasterio import features
import geopandas as gpd

from osgeo import gdal, ogr
import numpy as np


def generateHeatmap(dataframe, x_col, y_col, value_col, path_saving):
    def _generateBaseMap(default_location, default_zoom_start=12):
        return folium.Map(
            location=default_location, control_scale=True, zoom_start=default_zoom_start
        )

    default_location = (dataframe[x_col].mean(axis=0), dataframe[y_col].mean(axis=0))
    base_map = _generateBaseMap(default_location=default_location)

    HeatMap(
        data=dataframe[[x_col, y_col, value_col]]
        .groupby([x_col, y_col])
        .sum()
        .reset_index()
        .values.tolist(),
        radius=8,
        max_zoom=13,
    ).add_to(base_map)

    base_map.save(path_saving)


class GetRasterFromFile:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _load_data(self, tile):
        lon = np.load(os.path.join(root_dir, os.path.join(tile, "longitude.npy")))
        lat = np.load(os.path.join(root_dir, os.path.join(tile, "latitude.npy")))
        spectra = np.load(os.path.join(root_dir, os.path.join(tile, "training_X.npy")))
        class_ = np.load(os.path.join(root_dir, os.path.join(tile, "training_y.npy")))
        return lon, lat, spectra, class_

    def _get_bbox(self, lon, lat):
        xmin, xmax, ymin, ymax = np.min(lon), np.max(lon), np.min(lat), np.max(lat)
        bbox = sentinelhub.BBox(bbox=[(xmin, ymin), (xmax, ymax)], crs="EPSG:4326")
        utm_crs = str(CRS.get_utm_from_wgs84(xmin, ymin))
        bbox = sentinelhub.geo_utils.to_utm_bbox(bbox)
        return bbox, utm_crs

    def _vectorize(self, tile):

        lon, lat, spectra, class_ = self._load_data(tile)
        self.bbox, self.utm_crs = self._get_bbox(lon, lat)

        gdf = gpd.GeoDataFrame()
        gdf["geometry"] = [Point(x, y) for x, y in zip(lon, lat)]
        for i in range(spectra.shape[0]):
            gdf[f"B {i}"] = spectra[:, i]

        gdf["Class"] = class_
        gdf.crs = "EPSG:4326"

        gdf = gdf.to_crs(self.utm_crs)
        # Open the shapefile
        if not os.path.exists(
            os.path.join(root_dir, os.path.join(tile, "geodataframe"))
        ):
            os.mkdir(os.path.join(root_dir, os.path.join(tile, "geodataframe")))

        gdf.to_file(
            os.path.join(root_dir, os.path.join(tile, "geodataframe/geodataframe.shp")),
            index=False,
        )

        return gdf

    def execute(self, tile, field="Class"):

        gdf = self._vectorize(tile)
        fields = [k for k in gdf.columns if k != "geometry"]
        if field not in fields:
            raise ValueError(f"You must provide a field from the list {fields}")

        xmin, xmax, ymin, ymax = self.bbox

        # calculate size/resolution of the raster.
        x_res = int((xmax - xmin) / 10)
        y_res = int((ymax - ymin) / 10)
        geo_transform = (xmin, 10, 0, ymax, 0, -10)

        shapefile = ogr.Open(
            os.path.join(root_dir, os.path.join(tile, "geodataframe/geodataframe.shp"))
        )
        layer = shapefile.GetLayer()

        # Create a new raster
        options = [f"ATTRIBUTE={field}"]
        raster = gdal.GetDriverByName("GTiff").Create(
            os.path.join(root_dir, os.path.join(tile, f"{field}.tiff")),
            x_res,
            y_res,
            1,
            gdal.GDT_Float32,
        )

        raster.SetProjection(layer.GetSpatialRef().ExportToWkt())
        raster.SetGeoTransform(geo_transform)

        # Rasterize the shapefile
        gdal.RasterizeLayer(raster, [1], layer, burn_values=[1], options=options)

        # Close the raster and the shapefile
        raster = None
        shapefile = None
        # gdal_translate -co COMPRESS=LZW path/to/output.tiff path/to/compressed_output.tiff
        # remove output_fiff
