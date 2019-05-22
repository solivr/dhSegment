#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
from glob import glob
import os
import shutil


class BestExporterWithCheckpoints(tf.estimator.BestExporter):
    # Adapted from https://github.com/tensorflow/tensorflow/issues/8658#issuecomment-471415352
    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        export_result = None

        if self._model_dir != estimator.model_dir and self._event_file_pattern:
            # Loads best metric from event files.
            tf.logging.info('Loading best metric from event files.')

            self._model_dir = estimator.model_dir
            full_event_file_pattern = os.path.join(self._model_dir,
                                                   self._event_file_pattern)
            self._best_eval_result = self._get_best_eval_result(
                full_event_file_pattern)

        if self._best_eval_result is None or self._compare_fn(
                best_eval_result=self._best_eval_result,
                current_eval_result=eval_result):
            tf.logging.info('Performing best model export.')
            self._best_eval_result = eval_result
            export_result = self._saved_model_exporter.export(
                estimator, export_path, checkpoint_path, eval_result,
                is_the_final_export)  # export_result has byte format

            # copy the checkpoints files *.meta *.index, *.data* and checkpoint each time there is a better result,
            export_checkpoint_dir = os.path.join(export_result.decode(), 'checkpoint-data')
            os.makedirs(export_checkpoint_dir)
            for name in glob(estimator.latest_checkpoint() + '.*'):
                shutil.copy(name, os.path.join(export_checkpoint_dir, os.path.basename(name)))
            shutil.copy(os.path.join(export_path, os.pardir, 'checkpoint'),
                        os.path.join(export_checkpoint_dir, 'checkpoint'))

            self._garbage_collect_exports(export_path)

        return export_result
