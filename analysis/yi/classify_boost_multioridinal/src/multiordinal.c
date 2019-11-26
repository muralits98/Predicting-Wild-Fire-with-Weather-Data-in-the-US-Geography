#include <Python.h>
#include <numpy/arrayobject.h>

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

struct module_state {
    PyObject* error;
};

static PyObject* transform(PyObject* self, PyObject* args) {
    static const char* rain[15] = {
        "proximity shower rain",
        "proximity moderate rain",
        "light intensity drizzle",
        "light intensity drizzle rain",
        "drizzle",
        "shower drizzle",
        "heavy intensity drizzle",
        "light intensity shower rain",
        "ragged shower rain",
        "shower rain",
        "heavy intensity shower rain",
        "light rain",
        "moderate rain",
        "heavy intensity rain",
        "very heavy rain"
    };
    static const char* thunderstorm[13] = {
        "proximity thunderstorm",
        "proximity thunderstorm with drizzle",
        "proximity thunderstorm with rain",
        "ragged thunderstorm",
        "thunderstorm",
        "thunderstorm with light drizzle",
        "thunderstorm with drizzle",
        "thunderstorm with heavy drizzle",
        "thunderstorm with light rain",
        "thunderstorm with rain",
        "thunderstorm with heavy rain",
        "heavy thunderstorm",
        "tornado"
    };
    static const char* snow[7] = {
        "freezing rain",
        "light snow",
        "snow",
        "light shower snow",
        "shower snow",
        "heavy shower snow",
        "heavy snow"
    };
    static const char* cloud[4] = {
        "few clouds",
        "broken clouds",
        "scattered clouds",
        "overcast clouds"
    };
    static const char* rain_snow[4] = {
        "light shower sleet",
        "sleet",
        "light rain and snow",
        "rain and snow"
    };
    static const char* sand[3] = {
        "proximity sand/dust whirls",
        "sand",
        "sand/dust whirls"
    };
    static const char* misc[8] = {
        "sky is clear",
        "haze",
        "mist",
        "fog",
        "smoke",
        "squalls",
        "dust",
        "volcanic ash"
    };
    PyArrayObject *in_array;
    NpyIter *in_iter, *out_iter;
    NpyIter_IterNextFunc *in_iternext, *out_iter_next;
    PyObject ***in_data_ptr, *out_array;
    double *out_data, *out_data_ptr, **out_array_data_ptr;
    npy_intp *in_shape;
    
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &in_array,
                                        &PyArray_Type, &out_array)) {
        return NULL;
    }
    if (!(in_shape = PyArray_SHAPE(in_array))) {
        return NULL;
    } 
    if (!(out_data = calloc(54 * (*in_shape), sizeof(double)))) {
        return NULL;
    }
    if (!(in_iter = NpyIter_New(in_array, NPY_ITER_READONLY | NPY_ITER_REFS_OK,
                                NPY_KEEPORDER,
                                NPY_NO_CASTING, NULL))) {
        return NULL;
    }
    if (!(in_iternext = NpyIter_GetIterNext(in_iter, NULL))) {
        NpyIter_Deallocate(in_iter);
        return NULL;
    }
    in_data_ptr = (PyObject***)NpyIter_GetDataPtrArray(in_iter);
    out_data_ptr = out_data;
    do {
        double* temp = out_data_ptr;
        const char* data = PyUnicode_DATA(**in_data_ptr);
        int ind;
        for (ind = 0; ind < 15 && strcmp(data, rain[ind]); ind++);
        if (ind < 15) {
            for (int i = 0; i < ind + 1; i++, *(temp++) = 1);
            out_data_ptr += 54;
            continue;
        }
        temp += 15;
        for (ind = 0; ind < 13 && strcmp(data, thunderstorm[ind]); ind++);
        if (ind < 13) {
            for (int i = 0; i < ind + 1; i++, *(temp++) = 1);
            out_data_ptr += 54;
            continue;
        }
        temp += 13;
        for (ind = 0; ind < 7 && strcmp(data, snow[ind]); ind++);
        if (ind < 7) {
            for (int i = 0; i < ind + 1; i++, *(temp++) = 1);
            out_data_ptr += 54;
            continue;
        }
        temp += 7;
        for (ind = 0; ind < 4 && strcmp(data, cloud[ind]); ind++);
        if (ind < 4) {
            for (int i = 0; i < ind + 1; i++, *(temp++) = 1);
            out_data_ptr += 54;
            continue;
        }
        temp += 4;
        for (ind = 0; ind < 4 && strcmp(data, rain_snow[ind]); ind++);
        if (ind < 4) {
            for (int i = 0; i < ind + 1; i++, *(temp++) = 1);
            out_data_ptr += 54;
            continue;
        }
        temp += 4;
        for (ind = 0; ind < 3 && strcmp(data, sand[ind]); ind++);
        if (ind < 3) {
            for (int i = 0; i < ind + 1; i++, *(temp++) = 1);
            out_data_ptr += 54;
            continue;
        }
        temp += 3;
        for (ind = 0; ind < 8 && strcmp(data, misc[ind]); ind++);
        if (ind < 8) {
            temp[ind] = 1;
            out_data_ptr += 54;
        }
        temp += 8;
    } while (in_iternext(in_iter));

    NpyIter_Deallocate(in_iter);
    if (!(out_iter = NpyIter_New((PyArrayObject*)out_array, NPY_ITER_READWRITE,
                                 NPY_KEEPORDER,
                                 NPY_NO_CASTING, NULL))) {
        return NULL;
    }
    if (!(out_iter_next = NpyIter_GetIterNext(out_iter, NULL))) {
        NpyIter_Deallocate(out_iter);
        return NULL;
    }
    out_array_data_ptr = (double**)NpyIter_GetDataPtrArray(out_iter);
    out_data_ptr = out_data;
    do {
        **(out_array_data_ptr) = *(out_data_ptr++);
    } while (out_iter_next(out_iter));
    NpyIter_Deallocate(out_iter);
    free(out_data);
    Py_RETURN_NONE;
}

static PyMethodDef multiordinal_methods[] = {
    {"transform", (PyCFunction)transform, METH_VARARGS, 0},
    {0}
};

static int multiordinal_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int multiordinal_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "multiordinal",
    NULL,
    sizeof(struct module_state),
    multiordinal_methods,
    NULL,
    multiordinal_traverse,
    multiordinal_clear,
    NULL
};

PyMODINIT_FUNC PyInit_multiordinal(void) {
    PyObject* module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    import_array();
    return module;
}
