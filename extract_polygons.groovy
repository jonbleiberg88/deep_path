def top_level_path = buildFilePath(PROJECT_BASE_DIR, 'annotation_csv_files')

mkdirs(top_level_path)

def image_name = getProjectEntry().getImageName()

def annotation_objects = getAnnotationObjects()
def counter = 0
def annotation_classes = [:]

for (annotation_object in annotation_objects) {

    if (annotation_object.getPathClass() == null) {
        println("No class specified for Annotation $counter in $image_name")
        continue

    }

    name = counter.toString() + '.csv'
    def class_name = annotation_object.getPathClass().getName()
    if (annotation_classes.containsKey(class_name)) {
        annotation_dir = annotation_classes.get(class_name)
    } else {
        annotation_dir = create_class_dir(top_level_path, class_name, image_name)
        annotation_classes.put(class_name, annotation_dir)
    }

    annotation_file = buildFilePath(annotation_dir, name)
    def file = new File(annotation_file)
    file.text = ''

    def roi = annotation_object.getROI()

    for (point in roi.getPolygonPoints()) {
        file << point.getX() << ',' << point.getY() << System.lineSeparator()
    }
    counter++


}

def create_class_dir(top_level_path, path_class, image_name) {
    def class_name = path_class.replaceAll(" ", "_").toLowerCase()
    class_path = buildFilePath(top_level_path, (class_name +'_csv_files'), image_name)
    mkdirs(class_path)

    return class_path

}
