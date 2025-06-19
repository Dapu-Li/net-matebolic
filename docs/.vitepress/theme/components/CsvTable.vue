<script setup>
const props = defineProps({
  src: {
    type: String,
    required: true
  },
  label: {
    type: String,
    default: ''
  }
})

const openCsvViewer = () => {
  const encodedPath = encodeURIComponent(props.src)
  window.open(`/csv-viewer.html?src=${encodedPath}`, '_blank')
}
</script>

<template>
  <button @click="openCsvViewer" style="margin-bottom: 1em;">
    {{ label }}
  </button>
</template>
