def polygon_dir = buildFilePath(PROJECT_BASE_DIR, 'annotation_csv_files')
def sa_dir = buildFilePath(PROJECT_BASE_DIR, 'annotation_surface_areas')

mkdirs(polygon_dir)
mkdirs(sa_dir)


def image_name = getProjectEntry().getImageName()

slide_file_name = image_name + '.csv'
server = getCurrentImageData().getServer()

img_file = buildFilePath(sa_dir, slide_file_name)
def sa_file = new File(img_file)
sa_file.text = ''

sa_file << server.getHeight().toString() << ',' << server.getWidth().toString() << System.lineSeparator()

def annotation_objects = getAnnotationObjects()
def counter = 0
def annotation_classes = [:]

for (annotation_object in annotation_objects) {

    if (annotation_object.getPathClass() == null) {
        println("No class specified for Annotation $counter in $image_name")
        continue

    }

    name = counter.toString() + '.csv'
    def class_name = annotation_object.getPathClass().getName().replaceAll(" ", "_").toLowerCase()
    if (annotation_classes.containsKey(class_name)) {
        annotation_dir = annotation_classes.get(class_name)
    } else {
        annotation_dir = create_class_dir(polygon_dir, class_name, image_name)
        annotation_classes.put(class_name, annotation_dir)
    }

    annotation_file = buildFilePath(annotation_dir, name)
    def file = new File(annotation_file)
    file.text = ''

    def roi = annotation_object.getROI()

    for (point in roi.getPolygonPoints()) {
        file << point.getX() << ',' << point.getY() << System.lineSeparator()
    }

    def area = roi.getArea()
    sa_file << counter.toString() << ',' << class_name << ',' << area.toString() << System.lineSeparator()
    counter++


}

def create_class_dir(polygon_dir, class_name, image_name) {
    class_path = buildFilePath(polygon_dir, (class_name +'_csv_files'), image_name)
    mkdirs(class_path)

    return class_path

}
