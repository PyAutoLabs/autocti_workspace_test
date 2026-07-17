import autocti as ac

clocker = ac.Clocker2D(iterations=5, parallel_express=5, serial_express=5)

clocker.output_to_json(file_path="cti_clocker.json")

# ac.Clocker2D.from_json(file_path="cti_clocker.json")
